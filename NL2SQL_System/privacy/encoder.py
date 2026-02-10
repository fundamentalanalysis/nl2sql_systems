import hashlib
from typing import Dict, List, Tuple, Any
from privacy.config import encrypt_value
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from loguru import logger

# Global analyzer placeholder for lazy loading
_analyzer = None

# Entities we care about for masking
PII_ENTITIES = [
    "PERSON",
    "ORGANIZATION",
    "LOCATION",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "IN_AADHAAR",
]

def get_analyzer():
    """Lazy load the Presidio analyzer and handle missing models gracefully."""
    global _analyzer
    if _analyzer is None:
        try:
            # Prefer large English model if available
            nlp_configuration = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
            }
            provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
            nlp_engine = provider.create_engine()

            registry = RecognizerRegistry()
            registry.load_predefined_recognizers()

            # Add a lightweight ORG recognizer to catch common org suffixes
            org_pattern = Pattern(
                name="org_suffix",
                regex=r"\b[A-Z][A-Za-z&.'-]+(?:\s+[A-Z][A-Za-z&.'-]+)*\s+(Group|Ltd|Limited|LLC|Inc|Corp|Corporation|Company|Co\.|Technologies|Solutions|Systems|Enterprises)\b",
                score=0.6,
            )
            registry.add_recognizer(
                PatternRecognizer(
                    supported_entity="ORGANIZATION",
                    patterns=[org_pattern],
                )
            )

            # Aadhaar number (India) recognizer
            aadhaar_pattern = Pattern(
                name="aadhaar_number",
                regex=r"\b\d{4}\s?\d{4}\s?\d{4}\b",
                score=0.6,
            )
            registry.add_recognizer(
                PatternRecognizer(
                    supported_entity="IN_AADHAAR",
                    patterns=[aadhaar_pattern],
                )
            )

            _analyzer = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)
            logger.info("Presidio AnalyzerEngine loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load Presidio AnalyzerEngine (switching to regex fallback): {e}")
            _analyzer = "fallback"
    return _analyzer

def _regex_detect_pii(text: str) -> List[Dict[str, Any]]:
    """Simple regex fallback for PII detection when models are missing."""
    import re
    patterns = {
        "PHONE_NUMBER": r"\+?[\d\s-]{10,}",
        "EMAIL_ADDRESS": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "PERSON": r"\b[A-Z][a-z]+ [A-Z][a-z]+\b",  # Very basic name-like pattern
        "ORGANIZATION": r"\b[A-Z][A-Za-z&.'-]+(?:\s+[A-Z][A-Za-z&.'-]+)*\s+(Group|Ltd|Limited|LLC|Inc|Corp|Corporation|Company|Co\.|Technologies|Solutions|Systems|Enterprises)\b",
        "IN_AADHAAR": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    }
    
    entities = []
    for entity_type, pattern in patterns.items():
        for match in re.finditer(pattern, text):
            entities.append({
                "entity_type": entity_type,
                "start": match.start(),
                "end": match.end(),
                "score": 0.5
            })
    return entities

# In-memory store for tokens -> encrypted_values
# Format: { "TOKEN_HASH": "FERNET_ENCRYPTED_ORIGINAL_VALUE" }
_token_store: Dict[str, str] = {}

def detect_pii(text: str) -> List[Dict[str, Any]]:
    """
    Detect PII in text using Presidio (with regex fallback).
    Returns a list of dicts with entity_type, start, end, and score.
    """
    analyzer = get_analyzer()
    
    if analyzer == "fallback":
        return _regex_detect_pii(text)
        
    try:
        results = analyzer.analyze(text=text, language='en', entities=PII_ENTITIES)
        pii_entities = []
        for res in results:
            pii_entities.append({
                "entity_type": res.entity_type,
                "start": res.start,
                "end": res.end,
                "score": res.score
            })
        logger.info(f"Detected {len(pii_entities)} PII entities")
        return pii_entities
    except Exception as e:
        logger.error(f"PII Detection failed: {e}")
        return _regex_detect_pii(text)

def get_token_hash(value: str, entity_type: str) -> str:
    """Generate a unique but safe-looking token using SHA-256."""
    hash_obj = hashlib.sha256(value.encode())
    # Take first 8 chars for the token to keep it manageable
    short_hash = hash_obj.hexdigest()[:8].upper()
    return f"[{entity_type}_{short_hash}]"

def encode_query(text: str) -> Tuple[str, List[Dict]]:
    """
    Analyzes text, replaces PII with tokens, and stores encrypted maps.
    Returns: (encoded_text, list_of_mappings)
    """
    global _token_store
    results = detect_pii(text)
    
    # Sort results by start position in reverse to avoid index shifting
    results.sort(key=lambda x: x['start'], reverse=True)
    
    encoded_text = text
    mappings = []
    
    for res in results:
        start, end = res['start'], res['end']
        original_value = text[start:end]
        entity_type = res['entity_type']
        
        token = get_token_hash(original_value, entity_type)
        encrypted_val = encrypt_value(original_value)
        
        # Store in bridge
        _token_store[token] = encrypted_val
        
        # Replace in text
        encoded_text = encoded_text[:start] + token + encoded_text[end:]
        
        mappings.append({
            "token": token,
            "entity_type": entity_type,
            "original_len": end - start
        })
        
    return encoded_text, mappings

def encode_results(columns: List[str], rows: List[List[Any]]) -> List[List[Any]]:
    """
    Scans query results for PII and replaces with tokens.
    """
    encoded_rows = []
    for row in rows:
        encoded_row = []
        for val in row:
            if isinstance(val, str):
                # Use detect_pii on the string
                entities = detect_pii(val)
                if entities:
                    # Sort in reverse to replace
                    entities.sort(key=lambda x: x['start'], reverse=True)
                    ev = val
                    for ent in entities:
                        token = get_token_hash(val[ent['start']:ent['end']], ent['entity_type'])
                        encrypted_val = encrypt_value(val[ent['start']:ent['end']])
                        _token_store[token] = encrypted_val
                        ev = ev[:ent['start']] + token + ev[ent['end']:]
                    encoded_row.append(ev)
                else:
                    encoded_row.append(val)
            else:
                encoded_row.append(val)
        encoded_rows.append(encoded_row)
    return encoded_rows

def get_encrypted_mapping() -> Dict[str, str]:
    """Expose the current token store for potential persistence."""
    return _token_store
