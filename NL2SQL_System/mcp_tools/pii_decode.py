"""MCP Tool: pii_decode - Restore PII from tokens to original values."""
from typing import Dict, Any, Optional
from privacy.decoder import decode_text
from loguru import logger
from utils.trace_logger import emit_trace_event, start_timer, elapsed_ms


def pii_decode(text: str, trace_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Decode tokens in text back to their original sensitive values.
    
    Args:
        text: Text containing tokens (e.g., [PERSON_XXXX]) to be decoded
        
    Returns:
        The text with all tokens restored to original values.
    """
    trace_id = trace_id or "no-trace"
    timer = start_timer()
    logger.info("MCP Tool: pii_decode invoked")
    emit_trace_event(
        event="TOOL_INPUT",
        trace_id=trace_id,
        tool="pii_decode_tool",
        payload={"masked_text": text},
    )
    decoded_text = decode_text(text)
    output = {
        "decoded_text": decoded_text
    }
    emit_trace_event(
        event="TRANSFORM",
        trace_id=trace_id,
        tool="pii_decode_tool",
        payload={
            "name": "decode_text",
            "masked_text": text,
            "decoded_text": decoded_text,
        },
    )
    emit_trace_event(
        event="TOOL_OUTPUT",
        trace_id=trace_id,
        tool="pii_decode_tool",
        payload={"result": output, "execution_time_ms": elapsed_ms(timer)},
    )
    return output
