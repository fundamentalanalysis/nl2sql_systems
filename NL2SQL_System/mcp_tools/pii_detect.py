"""MCP Tool: pii_detect - Detect PII in text using high-accuracy patterns."""
from typing import Dict, Any, List
from privacy.encoder import detect_pii
from loguru import logger

def pii_detect(text: str) -> Dict[str, Any]:
    """
    Detect sensitive information in a string.
    
    Args:
        text: The text to analyze for PII
        
    Returns:
        List of detected entities with their types and locations.
    """
    logger.info("MCP Tool: pii_detect invoked")
    results = detect_pii(text)
    return {"entities": results, "count": len(results)}

if __name__ == "__main__":
    test_text = "My name is Jyothika and my phone is 6300769676. I work at Google."
    print(pii_detect(test_text))
