"""Structured trace logger for end-to-end AI observability."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from loguru import logger


TRACE_LOGGER_NAME = "ai_trace"
MAX_STRING_LEN = 240
MAX_LIST_ITEMS = 5
MAX_DICT_ITEMS = 30

# Keys that commonly carry huge payloads; log a compact summary instead.
HEAVY_KEYS = {
    "schema",
    "schema_info",
    "tables",
    "rows",
    "input_rows",
    "output_rows",
    "messages",
    "results_preview",
    "response_content",
}
DEFAULT_TRACE_MODE = os.getenv("TRACE_LOG_MODE", "minimal").strip().lower()
MINIMAL_EVENTS = {"QUERY_START", "QUERY_DONE", "TOOL_INPUT", "TOOL_OUTPUT", "TOOL_ERROR"}


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    """Recursively convert objects into JSON-safe structures."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, str) and len(value) > MAX_STRING_LEN:
            return f"{value[:MAX_STRING_LEN]}...(truncated)"
        return value
    if isinstance(value, dict):
        items = list(value.items())
        safe_dict = {}
        for idx, (k, v) in enumerate(items):
            if idx >= MAX_DICT_ITEMS:
                safe_dict["__extra_keys__"] = len(items) - MAX_DICT_ITEMS
                break
            key = str(k)
            if key in HEAVY_KEYS:
                safe_dict[key] = _summarize_heavy_value(v)
            else:
                safe_dict[key] = _json_safe(v)
        return safe_dict
    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        safe_list = [_json_safe(v) for v in seq[:MAX_LIST_ITEMS]]
        if len(seq) > MAX_LIST_ITEMS:
            safe_list.append(f"...({len(seq) - MAX_LIST_ITEMS} more)")
        return safe_list
    return str(value)


def _summarize_heavy_value(value: Any) -> Any:
    """Return concise summaries for commonly large values."""
    if isinstance(value, str):
        if len(value) > MAX_STRING_LEN:
            return {
                "type": "text",
                "length": len(value),
                "preview": f"{value[:MAX_STRING_LEN]}...(truncated)",
            }
        return value
    if isinstance(value, list):
        return {
            "type": "list",
            "count": len(value),
            "preview": _json_safe(value[:2]),
        }
    if isinstance(value, dict):
        # Special-case schema-like objects.
        tables = value.get("tables")
        if isinstance(tables, list):
            table_names = []
            for t in tables[:5]:
                if isinstance(t, dict):
                    table_names.append(t.get("name", "<unknown>"))
                else:
                    table_names.append(str(t))
            return {
                "type": "schema",
                "table_count": len(tables),
                "tables_preview": table_names,
            }
        return {
            "type": "object",
            "keys": list(value.keys())[:10],
            "key_count": len(value.keys()),
        }
    return _json_safe(value)


def emit_trace_event(
    event: str,
    trace_id: str,
    tool: str,
    payload: Optional[Dict[str, Any]] = None,
    level: str = "info",
) -> None:
    """
    Emit a structured JSON trace event suitable for audit pipelines.

    Args:
        event: Event type such as TOOL_INPUT / LLM_PROMPT / TOOL_OUTPUT
        trace_id: Correlation id spanning an end-to-end request
        tool: Logical tool/component name
        payload: Structured event payload
        level: loguru level to emit at
    """
    if DEFAULT_TRACE_MODE == "minimal" and event not in MINIMAL_EVENTS:
        return

    event_data = {
        "logger": TRACE_LOGGER_NAME,
        "event": event,
        "trace_id": trace_id,
        "tool": tool,
        "timestamp": _utc_timestamp(),
        "payload": _json_safe(payload or {}),
    }
    message = json.dumps(event_data, default=str)
    trace_logger = logger.bind(trace_event=True, trace_id=trace_id, trace_tool=tool)
    if level == "error":
        trace_logger.error(message)
    elif level == "warning":
        trace_logger.warning(message)
    elif level == "debug":
        trace_logger.debug(message)
    else:
        trace_logger.info(message)


def start_timer() -> float:
    """Start a high-precision timer."""
    return time.perf_counter()


def elapsed_ms(started_at: float) -> float:
    """Elapsed duration in milliseconds."""
    return round((time.perf_counter() - started_at) * 1000.0, 3)
