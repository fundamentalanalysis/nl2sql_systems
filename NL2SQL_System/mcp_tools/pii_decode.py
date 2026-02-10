"""MCP Tool: pii_decode - Restore PII from tokens to original values."""
from typing import Dict, Any
from privacy.decoder import decode_text
from loguru import logger

def pii_decode(text: str) -> Dict[str, Any]:
    """
    Decode tokens in text back to their original sensitive values.
    
    Args:
        text: Text containing tokens (e.g., [PERSON_XXXX]) to be decoded
        
    Returns:
        The text with all tokens restored to original values.
    """
    logger.info("MCP Tool: pii_decode invoked")
    decoded_text = decode_text(text)
    return {
        "decoded_text": decoded_text
    }

if __name__ == "__main__":
    # This requires state in encoder._token_store
    print(pii_decode("The manager is [PERSON_1234]"))
