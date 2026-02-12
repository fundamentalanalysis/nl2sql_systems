"""MCP Tool: pii_detect - Detect PII in text using high-accuracy patterns."""
from typing import Dict, Any, Optional
from privacy.encoder import detect_pii
from loguru import logger
from utils.trace_logger import emit_trace_event, start_timer, elapsed_ms


def pii_detect(text: str, trace_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Detect sensitive information in a string.
    
    Args:
        text: The text to analyze for PII
        
    Returns:
        List of detected entities with their types and locations.
    """
    trace_id = trace_id or "no-trace"
    timer = start_timer()
    logger.info("MCP Tool: pii_detect invoked")
    emit_trace_event(
        event="TOOL_INPUT",
        trace_id=trace_id,
        tool="pii_detect_tool",
        payload={"text": text},
    )
    results = detect_pii(text)
    output = {"entities": results, "count": len(results)}
    emit_trace_event(
        event="PII_DETECTED",
        trace_id=trace_id,
        tool="pii_detect_tool",
        payload=output,
    )
    emit_trace_event(
        event="TOOL_OUTPUT",
        trace_id=trace_id,
        tool="pii_detect_tool",
        payload={"result": output, "execution_time_ms": elapsed_ms(timer)},
    )
    return output
