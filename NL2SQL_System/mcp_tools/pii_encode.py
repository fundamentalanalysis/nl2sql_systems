"""MCP Tool: pii_encode - Replace PII with safe SHA-256 tokens."""
from typing import Dict, Any, Optional
from privacy.encoder import encode_query
from loguru import logger
from utils.trace_logger import emit_trace_event, start_timer, elapsed_ms


def pii_encode(text: str, trace_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Encode PII in text using placeholder tokenization.
    Replaces original values with tokens like [PERSON_XXXX] and secures values in memory.
    
    Args:
        text: The text containing PII to be encoded
        
    Returns:
        The encoded text and metadata about mappings.
    """
    trace_id = trace_id or "no-trace"
    timer = start_timer()
    logger.info("MCP Tool: pii_encode invoked")
    emit_trace_event(
        event="TOOL_INPUT",
        trace_id=trace_id,
        tool="pii_encode_tool",
        payload={"original_text": text},
    )
    encoded_text, mappings = encode_query(text)
    output = {
        "encoded_text": encoded_text,
        "mappings": mappings,
        "count": len(mappings)
    }
    emit_trace_event(
        event="PII_MASKED",
        trace_id=trace_id,
        tool="pii_encode_tool",
        payload={
            "original_text": text,
            "masked_text": encoded_text,
            "mappings": mappings,
            "count": len(mappings),
        },
    )
    emit_trace_event(
        event="TOOL_OUTPUT",
        trace_id=trace_id,
        tool="pii_encode_tool",
        payload={"result": output, "execution_time_ms": elapsed_ms(timer)},
    )
    return output
