import hashlib
import json
import os
import threading
import re
from typing import Dict, List, Tuple, Any
from privacy.config import encrypt_value
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from loguru import logger
from app.services.redis import redis_client

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

_PERSON_SUFFIX_STOPWORDS = {
    "and", "or", "of", "to", "for", "from", "in", "on", "at", "by", "with",
    "details", "detail", "show", "get", "give", "find", "list", "about",
    "belongs", "belonging", "customer", "customers", "order", "orders",
    "email", "emails", "phone", "phones", "address", "addresses",
}

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
_PII_TTL_SECONDS = 86400
_TOKEN_FILE = os.path.join(os.path.dirname(__file__), ".token_store.json")
_TOKEN_FILE_LOCK = threading.Lock()

# Columns that are generally safe to skip for result-side PII detection.
_SAFE_RESULT_COLUMNS = {
    "postal_code",
    "zipcode",
    "zip_code",
    "country_code",
    "state_code",
    "item_status",
    "order_status",
}


def _is_low_risk_result_column(column_name: str) -> bool:
    col = (column_name or "").strip().lower()
    if not col:
        return False
    if col in _SAFE_RESULT_COLUMNS:
        return True
    if col.endswith("_id") or col.endswith("_count"):
        return True
    if col.startswith("is_") or col.startswith("has_"):
        return True
    return False


def _is_pii_candidate_text(value: str) -> bool:
    """
    Fast pre-filter to avoid expensive NLP on obvious non-PII strings.
    """
    if not value:
        return False
    s = value.strip()
    if not s:
        return False
    # Pure numeric / punctuation-like strings (postal codes, IDs, etc.)
    if not any(ch.isalpha() for ch in s):
        return False
    # Tiny tokens are unlikely to carry meaningful PII
    if len(s) < 3:
        return False
    return True


def _load_token_file() -> Dict[str, str]:
    if not os.path.exists(_TOKEN_FILE):
        return {}
    try:
        with _TOKEN_FILE_LOCK, open(_TOKEN_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _persist_token(token: str, encrypted_val: str) -> None:
    """Persist token mapping locally (encrypted values only)."""
    try:
        with _TOKEN_FILE_LOCK:
            data = _load_token_file()
            data[token] = encrypted_val
            with open(_TOKEN_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f)
    except Exception:
        # Best-effort persistence
        pass


def get_persisted_token(token: str) -> str:
    """Retrieve persisted token mapping if available."""
    data = _load_token_file()
    return data.get(token)

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
        pii_entities = _expand_person_entities(text, pii_entities)
        # Debug-level to avoid noisy logs in row-by-row result masking loops.
        logger.debug(f"Detected {len(pii_entities)} PII entities")
        return pii_entities
    except Exception as e:
        logger.error(f"PII Detection failed: {e}")
        return _expand_person_entities(text, _regex_detect_pii(text))


def _expand_person_entities(text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Post-process PERSON entities to capture likely full names when detector
    catches only the first token (e.g., 'erin' from 'erin white').
    """
    if not entities:
        return entities

    expanded: List[Dict[str, Any]] = []
    entities_sorted = sorted(entities, key=lambda x: (x.get("start", 0), x.get("end", 0)))

    for ent in entities_sorted:
        if ent.get("entity_type") != "PERSON":
            expanded.append(ent)
            continue

        start = ent.get("start")
        end = ent.get("end")
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end > len(text) or start >= end:
            expanded.append(ent)
            continue

        # If current PERSON already looks multi-token, keep as-is.
        current_value = text[start:end].strip()
        if len(current_value.split()) >= 2:
            expanded.append(ent)
            continue

        # Expand to include one adjacent surname-like token.
        m = re.match(r"\s+([A-Za-z][A-Za-z'\-]{1,29})\b", text[end:])
        if not m:
            expanded.append(ent)
            continue

        next_token = m.group(1)
        if next_token.lower() in _PERSON_SUFFIX_STOPWORDS:
            expanded.append(ent)
            continue

        new_end = end + m.end(1)
        updated = dict(ent)
        updated["end"] = new_end
        expanded.append(updated)

    # Deduplicate exact overlaps after expansion.
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for ent in expanded:
        key = (ent.get("entity_type"), ent.get("start"), ent.get("end"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ent)

    return deduped

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
        if redis_client.is_connected:
            redis_client.set_pii_mapping(token, encrypted_val, _PII_TTL_SECONDS)
        else:
            _persist_token(token, encrypted_val)
        
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
    safe_idx = {
        idx
        for idx, col in enumerate(columns or [])
        if _is_low_risk_result_column(col)
    }

    encoded_rows = []
    masked_cells = 0
    scanned_strings = 0
    for row in rows:
        encoded_row = []
        for idx, val in enumerate(row):
            if isinstance(val, str):
                scanned_strings += 1
                if idx in safe_idx or not _is_pii_candidate_text(val):
                    encoded_row.append(val)
                    continue
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
                        if redis_client.is_connected:
                            redis_client.set_pii_mapping(token, encrypted_val, _PII_TTL_SECONDS)
                        else:
                            _persist_token(token, encrypted_val)
                        ev = ev[:ent['start']] + token + ev[ent['end']:]
                    masked_cells += 1
                    encoded_row.append(ev)
                else:
                    encoded_row.append(val)
            else:
                encoded_row.append(val)
        encoded_rows.append(encoded_row)
    logger.info(
        f"Result PII masking summary: rows={len(rows)}, scanned_strings={scanned_strings}, masked_cells={masked_cells}"
    )
    return encoded_rows

def get_encrypted_mapping() -> Dict[str, str]:
    """Expose the current token store for potential persistence."""
    return _token_store
