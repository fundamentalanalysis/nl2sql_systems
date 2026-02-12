"""LangChain agent setup with tool calling."""
from typing import Dict, Any, List, Optional, Tuple
from langchain_openai import AzureChatOpenAI
from loguru import logger
from app.config import settings
from agent.tools import LANGCHAIN_TOOLS
from mcp_tools.get_schema import get_schema as mcp_get_schema
from mcp_tools.generate_sql import generate_sql as mcp_generate_sql
from mcp_tools.execute_sql import execute_sql as mcp_execute_sql
from mcp_tools.summarize_results import summarize_results as mcp_summarize_results
from mcp_tools.pii_detect import pii_detect as mcp_pii_detect
from mcp_tools.pii_encode import pii_encode as mcp_pii_encode
from mcp_tools.pii_decode import pii_decode as mcp_pii_decode
from privacy.decoder import decode_results
from database.connection import db_manager
import time
import json
from uuid import uuid4
from utils.trace_logger import emit_trace_event, start_timer, elapsed_ms


def _new_trace_id() -> str:
    return uuid4().hex[:10]


def _short(value: Any, limit: int = 220) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "...(truncated)"


def _extract_direct_answer_preview(text: str, limit: int = 260) -> str:
    s = (text or "").strip()
    marker = "1. **Direct Answer:**"
    idx = s.find(marker)
    if idx != -1:
        s = s[idx + len(marker):].strip()
    # Collapse newlines for cleaner single-line logs.
    s = " ".join(s.split())
    return _short(s, limit)


def _safe_tool_args(tool_args: Any) -> Any:
    if not isinstance(tool_args, dict):
        return _short(tool_args)
    safe = {}
    for key, value in tool_args.items():
        safe[key] = _short(value) if isinstance(value, str) else value
    return safe


def _tool_result_summary(tool_name: str, tool_result: Any, source_text: Optional[str] = None) -> str:
    if tool_name == "execute_sql_tool" and isinstance(tool_result, dict):
        row_count = tool_result.get("row_count")
        cols = tool_result.get("columns", [])
        return f"rows={row_count}, columns={len(cols)}"
    if tool_name == "pii_detect_tool" and isinstance(tool_result, dict):
        if source_text:
            return f"entities={tool_result.get('count', 0)}, values={_pii_entity_value_summary(source_text, tool_result)}"
        return f"entities={tool_result.get('count', 0)}, types={_pii_entity_type_summary(tool_result)}"
    if tool_name == "pii_encode_tool" and isinstance(tool_result, dict):
        return f"mappings={tool_result.get('count', 0)}, encoded_text={_short(tool_result.get('encoded_text', ''))}"
    if tool_name == "generate_sql_tool":
        return f"sql={tool_result}"
    if tool_name == "summarize_results_tool":
        text = str(tool_result)
        masked_tokens = text.count("[PERSON_") + text.count("[LOCATION_") + text.count("[EMAIL_ADDRESS_")
        return f"summary_len={len(text)}, masked_tokens={masked_tokens}"
    if tool_name == "pii_decode_tool":
        text = str(tool_result)
        remaining_tokens = text.count("[PERSON_") + text.count("[LOCATION_") + text.count("[EMAIL_ADDRESS_")
        return (
            f"decoded_len={len(text)}, tokens_remaining={remaining_tokens}, "
            f"direct_answer={_extract_direct_answer_preview(text)}"
        )
    return _short(tool_result)


def _pii_entity_type_summary(pii_result: Dict[str, Any]) -> str:
    entities = pii_result.get("entities", []) if isinstance(pii_result, dict) else []
    if not entities:
        return "none"

    counts: Dict[str, int] = {}
    for ent in entities:
        et = str(ent.get("entity_type", "UNKNOWN")).upper()
        counts[et] = counts.get(et, 0) + 1

    parts = [f"{et}({cnt})" for et, cnt in sorted(counts.items(), key=lambda x: x[0])]
    return ", ".join(parts)


def _pii_entity_value_summary(source_text: str, pii_result: Dict[str, Any]) -> str:
    entities = pii_result.get("entities", []) if isinstance(pii_result, dict) else []
    if not entities:
        return "none"

    values: List[str] = []
    for ent in entities:
        s = ent.get("start")
        e = ent.get("end")
        if isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= len(source_text):
            val = source_text[s:e].strip()
            if val:
                values.append(val)

    # Preserve order, remove duplicates (case-insensitive)
    seen = set()
    uniq = []
    for val in values:
        key = val.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(val)

    return ", ".join(uniq) if uniq else "none"


def _human_log(message: str) -> None:
    print(message, flush=True)


def _human_banner(title: str, lines: Optional[List[Tuple[str, str]]] = None) -> None:
    bar = "━" * 50
    _human_log(bar)
    _human_log(title)
    if lines:
        for key, value in lines:
            _human_log(f"   {key:<10}: {value}")
    _human_log(bar)


def _human_step(step_num: int, title: str, detail: Optional[str] = None) -> None:
    _human_log(f"\n{title}")
    if detail:
        _human_log(f"   -> {detail}")


def _schema_has_column(schema: Dict[str, Any], table: str, column: str) -> bool:
    for t in schema.get("tables", []):
        if t.get("name") == table:
            return any(c.get("name") == column for c in t.get("columns", []))
    return False


def _extract_entity_values(question: str, pii_detect_res: Dict[str, Any], entity_type: str) -> List[str]:
    values: List[str] = []
    entities = pii_detect_res.get("entities", []) if isinstance(pii_detect_res, dict) else []
    for ent in entities:
        if ent.get("entity_type") != entity_type:
            continue
        s = ent.get("start")
        e = ent.get("end")
        if isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= len(question):
            val = question[s:e].strip()
            if val:
                values.append(val)
    # Preserve order, remove duplicates
    seen = set()
    uniq = []
    for v in values:
        key = v.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(v)
    return uniq


def _count_column_matches(table: str, column: str, value: str) -> int:
    query = f"SELECT COUNT(*) AS c FROM {table} WHERE LOWER({column}) = LOWER(%s)"
    try:
        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (value,))
                row = cursor.fetchone() or {}
                return int(row.get("c", 0))
    except Exception:
        return 0


def _build_location_resolution_hint(
    question: str,
    pii_detect_res: Dict[str, Any],
    schema: Dict[str, Any],
    trace_id: str,
) -> Optional[str]:
    """
    Resolve LOCATION entity to best-fit customers column (state/country/city) by DB value checks.
    Returns a compact hint string for SQL generation or None.
    """
    location_values = _extract_entity_values(question, pii_detect_res, "LOCATION")
    emit_trace_event(
        event="TRANSFORM",
        trace_id=trace_id,
        tool="location_resolver",
        payload={
            "name": "extract_location_entities",
            "location_values": location_values,
        },
    )
    if not location_values:
        return None

    if not _schema_has_column(schema, "customers", "state") and not _schema_has_column(schema, "customers", "country") and not _schema_has_column(schema, "customers", "city"):
        return None

    target_cols = []
    for col in ("state", "country", "city"):
        if _schema_has_column(schema, "customers", col):
            target_cols.append(col)

    hints = []
    for loc in location_values:
        counts: Dict[str, int] = {}
        for col in target_cols:
            counts[col] = _count_column_matches("customers", col, loc)

        # Prefer highest non-zero match. If tie, prefer state > city > country for customer geo queries.
        best_col = None
        best_count = 0
        pref_order = {"state": 3, "city": 2, "country": 1}
        for col, cnt in counts.items():
            if cnt > best_count:
                best_col, best_count = col, cnt
            elif cnt == best_count and cnt > 0 and best_col is not None:
                if pref_order.get(col, 0) > pref_order.get(best_col, 0):
                    best_col = col

        if best_col and best_count > 0:
            logger.info(
                f"[trace={trace_id}] LOCATION_RESOLVER value={loc!r} chosen=customers.{best_col} counts={counts}"
            )
            emit_trace_event(
                event="INTERNAL_ACTION",
                trace_id=trace_id,
                tool="location_resolver",
                payload={
                    "value": loc,
                    "counts": counts,
                    "chosen_column": f"customers.{best_col}",
                    "matched_rows": best_count,
                },
            )
            hints.append(
                f"For location '{loc}', use customers.{best_col} (exact DB match found)."
            )
        else:
            logger.info(
                f"[trace={trace_id}] LOCATION_RESOLVER value={loc!r} no_exact_match counts={counts}"
            )
            emit_trace_event(
                event="INTERNAL_ACTION",
                trace_id=trace_id,
                tool="location_resolver",
                payload={
                    "value": loc,
                    "counts": counts,
                    "chosen_column": None,
                },
            )

    if not hints:
        return None
    hint = " LOCATION_COLUMN_HINTS: " + " ".join(hints)
    emit_trace_event(
        event="TRANSFORM",
        trace_id=trace_id,
        tool="location_resolver",
        payload={
            "name": "build_location_hint",
            "hint": hint,
        },
    )
    return hint


class MySQLAnalyticalAgent:
    """MySQL Analytical Agent using LangChain with tool calling."""

    def __init__(self):
        """Initialize the agent with LLM and tools."""
        logger.info("Initializing MySQL Analytical Agent")
        
        # Set Azure OpenAI API key as environment variable (required by langchain-openai)
        import os
        os.environ["OPENAI_API_KEY"] = settings.azure_ai_api_key

        # Initialize LLM with tools
        self.llm = AzureChatOpenAI(
            azure_endpoint=settings.azure_ai_endpoint,
            azure_deployment=settings.azure_openai_deployment,
            api_version=settings.azure_ai_api_version,
            temperature=0,
            timeout=180.0  # 3 minute timeout for complex analytical queries
        )

        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(LANGCHAIN_TOOLS)

        logger.info("Agent initialized successfully")

    def query(self, question: str, role: str = "admin") -> Dict[str, Any]:
        """Deterministic privacy-first pipeline (no planner LLM step)."""
        role = "admin"
        trace_id = _new_trace_id()
        start_time = time.time()
        timer = start_timer()
        steps = 7

        logger.info(f"[trace={trace_id}] QUERY_START role={role} question={_short(question)}")
        logger.info(f"[trace={trace_id}] PIPELINE User -> PII Detect -> PII Encode -> SQL LLM -> SQL Decode/Execute -> PII Encode Results -> Summary LLM -> PII Decode Final")
        emit_trace_event(
            event="QUERY_START",
            trace_id=trace_id,
            tool="agent",
            payload={"role": role, "question": question},
        )
        _human_banner(
            "🔵 New Query Received",
            [("Trace ID", trace_id), ("User Role", role), ("Question", question)],
        )

        try:
            sql_llm_time = 0.0
            db_exec_time = 0.0
            summary_llm_time = 0.0
            logger.info(f"[trace={trace_id}] TOOL_START step=1 tool=pii_detect_tool")
            _human_step(1, "🔎 Step 1: Detecting Sensitive Information (PII)")
            pii_detect_res = mcp_pii_detect(question, trace_id=trace_id)
            _human_log(
                f"   -> Found {pii_detect_res.get('count', 0)} sensitive entities "
                f"[{_pii_entity_value_summary(question, pii_detect_res)}]"
            )
            logger.info(
                f"[trace={trace_id}] TOOL_DONE step=1 tool=pii_detect_tool "
                f"result={_tool_result_summary('pii_detect_tool', pii_detect_res, question)}"
            )

            logger.info(f"[trace={trace_id}] TOOL_START step=2 tool=pii_encode_tool")
            _human_step(2, "🔐 Step 2: Protecting Sensitive Data")
            pii_encode_res = mcp_pii_encode(question, trace_id=trace_id)
            encoded_question = pii_encode_res.get("encoded_text", question)
            emit_trace_event(
                event="TRANSFORM",
                trace_id=trace_id,
                tool="agent",
                payload={
                    "name": "encoded_question",
                    "original_question": question,
                    "encoded_question": encoded_question,
                },
            )
            _human_log(f"   -> Masked query: {_short(encoded_question, 180)}")
            logger.info(f"[trace={trace_id}] TOOL_DONE step=2 tool=pii_encode_tool result={_tool_result_summary('pii_encode_tool', pii_encode_res)}")

            logger.info(f"[trace={trace_id}] TOOL_START step=3 tool=get_schema_tool")
            _human_step(3, "📚 Step 3: Loading Allowed Schema")
            schema = mcp_get_schema(role, trace_id=trace_id)
            _human_log(f"   -> Loaded {len(schema.get('tables', []))} tables")
            logger.info(f"[trace={trace_id}] TOOL_DONE step=3 tool=get_schema_tool tables={len(schema.get('tables', []))}")

            location_hint = _build_location_resolution_hint(question, pii_detect_res, schema, trace_id)
            sql_question = encoded_question + (location_hint or "")
            emit_trace_event(
                event="TRANSFORM",
                trace_id=trace_id,
                tool="agent",
                payload={
                    "name": "sql_input_question",
                    "encoded_question": encoded_question,
                    "location_hint": location_hint,
                    "sql_question": sql_question,
                },
            )
            logger.info(f"[trace={trace_id}] TOOL_START step=4 tool=generate_sql_tool args={{'question': {_short(sql_question)}}}")
            _human_step(4, "🧠 Step 4: Generating SQL using AI", "AI receives masked question only")
            t_sql_start = time.time()
            sql_res = mcp_generate_sql(sql_question, schema, trace_id=trace_id)
            sql_llm_time = time.time() - t_sql_start
            sql_query = sql_res["sql"]
            _human_log("   -> SQL query generated successfully")
            _human_log(f"   -> SQL: {sql_query}")
            _human_log(f"   -> SQL LLM response time: {sql_llm_time:.2f}s")
            logger.info(f"[trace={trace_id}] TOOL_DONE step=4 tool=generate_sql_tool sql={sql_query}")

            logger.info(f"[trace={trace_id}] TOOL_START step=5 tool=execute_sql_tool args={{'sql': {_short(sql_query)}, 'role': '{role}'}}")
            _human_step(5, "🗄️ Step 5: Executing Query in Database")
            t_db_start = time.time()
            query_res = mcp_execute_sql(sql_query, role, trace_id=trace_id)
            db_exec_time = time.time() - t_db_start
            _human_log(f"   -> Retrieved {query_res.get('row_count', 0)} records")
            _human_log(f"   -> DB execution time: {db_exec_time:.2f}s")
            logger.info(f"[trace={trace_id}] TOOL_DONE step=5 tool=execute_sql_tool rows={query_res.get('row_count', 0)} cols={len(query_res.get('columns', []))}")

            logger.info(f"[trace={trace_id}] TOOL_START step=6 tool=summarize_results_tool")
            _human_step(6, "🧠 Step 6: Generating Final Answer", f"AI summarizing {query_res.get('row_count', 0)} rows")
            t_summary_start = time.time()
            summary_res = mcp_summarize_results(
                question=encoded_question,
                columns=query_res["columns"],
                rows=query_res["rows"],
                row_count=query_res["row_count"],
                trace_id=trace_id,
            )
            summary_llm_time = time.time() - t_summary_start
            masked_summary = summary_res["summary"]
            _human_log(f"   -> Summary LLM response time: {summary_llm_time:.2f}s")
            logger.info(f"[trace={trace_id}] TOOL_DONE step=6 tool=summarize_results_tool summary_len={len(masked_summary)}")

            logger.info(f"[trace={trace_id}] TOOL_START step=7 tool=pii_decode_tool")
            _human_step(7, "🔓 Step 7: Restoring Sensitive Data for Final Output")
            final_answer = mcp_pii_decode(masked_summary, trace_id=trace_id)["decoded_text"]
            logger.info(f"[trace={trace_id}] TOOL_DONE step=7 tool=pii_decode_tool")
            emit_trace_event(
                event="QUERY_FINAL_OUTPUT",
                trace_id=trace_id,
                tool="agent",
                payload={
                    "masked_summary": masked_summary,
                    "final_answer": final_answer,
                },
            )

            execution_time = time.time() - start_time
            logger.info(f"[trace={trace_id}] QUERY_DONE status=success execution_time={round(execution_time,2)} steps={steps}")
            emit_trace_event(
                event="QUERY_DONE",
                trace_id=trace_id,
                tool="agent",
                payload={
                    "status": "success",
                    "execution_time_seconds": round(execution_time, 3),
                    "execution_time_ms": elapsed_ms(timer),
                    "steps": steps,
                },
            )
            _human_banner(
                "✅ Query Completed Successfully",
                [
                    ("Trace ID", trace_id),
                    ("Records", str(query_res.get("row_count", 0))),
                    ("SQL LLM", f"{sql_llm_time:.2f}s"),
                    ("DB Time", f"{db_exec_time:.2f}s"),
                    ("Summary LLM", f"{summary_llm_time:.2f}s"),
                    ("Exec Time", f"{round(execution_time, 2)} seconds"),
                ],
            )
            _human_banner(
                "🔐 Security Summary",
                [
                    ("PII Detected", str(pii_detect_res.get("count", 0))),
                    ("LLM Masked Input", "YES"),
                    ("SQL Restored for DB", "YES"),
                    ("Results Masked pre-LLM", "YES"),
                    ("Final Output Decoded", "YES"),
                ],
            )
            return {
                "answer": final_answer,
                "execution_time": round(execution_time, 2),
                "reasoning_steps": steps,
            }
        except Exception as e:
            execution_time = time.time() - start_time
            logger.exception(f"[trace={trace_id}] QUERY_DONE status=error execution_time={round(execution_time,2)} error={e}")
            emit_trace_event(
                event="QUERY_DONE",
                trace_id=trace_id,
                tool="agent",
                payload={
                    "status": "error",
                    "error": str(e),
                    "execution_time_seconds": round(execution_time, 3),
                    "execution_time_ms": elapsed_ms(timer),
                },
                level="error",
            )
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "execution_time": round(execution_time, 2),
                "reasoning_steps": 0,
                "error": str(e),
            }

    def query_stream(self, question: str, role: str = "admin"):
        """Streaming deterministic privacy-first pipeline (no planner LLM step)."""
        role = "admin"
        trace_id = _new_trace_id()
        start_time = time.time()
        timer = start_timer()

        step_names = {
            1: ("pii_detect_tool", "Analyzing Privacy Risk"),
            2: ("pii_encode_tool", "Masking Sensitive Data"),
            3: ("get_schema_tool", "Getting Database Schema"),
            4: ("generate_sql_tool", "Generating SQL Query"),
            5: ("execute_sql_tool", "Executing Query"),
            6: ("summarize_results_tool", "Analyzing Results"),
            7: ("pii_decode_tool", "Restoring Sensitive Data"),
        }

        def _emit_start(step: int):
            tool_name, step_name = step_names[step]
            logger.info(f"[trace={trace_id}] TOOL_START step={step} tool={tool_name}")
            return {
                "type": "step_start",
                "step_name": step_name,
                "tool_name": tool_name,
                "step_number": step,
            }

        def _emit_done(step: int, result: Any):
            tool_name, step_name = step_names[step]
            source_text = question if tool_name == "pii_detect_tool" else None
            logger.info(
                f"[trace={trace_id}] TOOL_DONE step={step} tool={tool_name} "
                f"result={_tool_result_summary(tool_name, result, source_text)}"
            )
            return {
                "type": "step_complete",
                "step_name": step_name,
                "tool_name": tool_name,
                "step_number": step,
                "status": "success",
                "tool_result": json.dumps(result) if isinstance(result, (dict, list)) else str(result),
            }

        logger.info(f"[trace={trace_id}] STREAM_START role={role} question={_short(question)}")
        logger.info(f"[trace={trace_id}] PIPELINE User -> PII Detect -> PII Encode -> SQL LLM -> SQL Decode/Execute -> PII Encode Results -> Summary LLM -> PII Decode Final")
        emit_trace_event(
            event="QUERY_START",
            trace_id=trace_id,
            tool="agent_stream",
            payload={"role": role, "question": question},
        )
        _human_banner(
            "🔵 New Query Received",
            [("Trace ID", trace_id), ("User Role", role), ("Question", question)],
        )

        try:
            last_query_results = None
            last_query_results_decoded = None
            query_res = {}
            sql_llm_time = 0.0
            db_exec_time = 0.0
            summary_llm_time = 0.0

            yield _emit_start(1)
            _human_step(1, "🔎 Step 1: Detecting Sensitive Information (PII)")
            pii_detect_res = mcp_pii_detect(question, trace_id=trace_id)
            _human_log(
                f"   -> Found {pii_detect_res.get('count', 0)} sensitive entities "
                f"[{_pii_entity_value_summary(question, pii_detect_res)}]"
            )
            pii_entities = pii_detect_res.get("entities", []) if isinstance(pii_detect_res, dict) else []
            pii_summary = {
                "count": pii_detect_res.get("count", 0) if isinstance(pii_detect_res, dict) else 0,
                "entities": [
                    {
                        "type": ent.get("entity_type"),
                        "value": question[ent.get("start", 0):ent.get("end", 0)],
                    }
                    for ent in pii_entities
                ],
            }
            yield _emit_done(1, pii_detect_res)

            yield _emit_start(2)
            _human_step(2, "🔐 Step 2: Protecting Sensitive Data")
            pii_encode_res = mcp_pii_encode(question, trace_id=trace_id)
            encoded_question = pii_encode_res.get("encoded_text", question)
            _human_log(f"   -> Masked query: {_short(encoded_question, 180)}")
            yield _emit_done(2, pii_encode_res)

            yield _emit_start(3)
            _human_step(3, "📚 Step 3: Loading Allowed Schema")
            schema = mcp_get_schema(role, trace_id=trace_id)
            _human_log(f"   -> Loaded {len(schema.get('tables', []))} tables")
            yield _emit_done(3, {"table_count": len(schema.get("tables", []))})

            yield _emit_start(4)
            _human_step(4, "🧠 Step 4: Generating SQL using AI", "AI receives masked question only")
            location_hint = _build_location_resolution_hint(question, pii_detect_res, schema, trace_id)
            sql_question = encoded_question + (location_hint or "")
            t_sql_start = time.time()
            sql_res = mcp_generate_sql(sql_question, schema, trace_id=trace_id)
            sql_llm_time = time.time() - t_sql_start
            sql_query = sql_res["sql"]
            _human_log("   -> SQL query generated successfully")
            _human_log(f"   -> SQL: {sql_query}")
            _human_log(f"   -> SQL LLM response time: {sql_llm_time:.2f}s")
            yield _emit_done(4, sql_query)

            yield _emit_start(5)
            _human_step(5, "🗄️ Step 5: Executing Query in Database")
            t_db_start = time.time()
            query_res = mcp_execute_sql(sql_query, role, trace_id=trace_id)
            db_exec_time = time.time() - t_db_start
            last_query_results = query_res
            last_query_results_decoded = decode_results(query_res)
            _human_log(f"   -> Retrieved {query_res.get('row_count', 0)} records")
            _human_log(f"   -> DB execution time: {db_exec_time:.2f}s")
            yield _emit_done(5, query_res)

            yield _emit_start(6)
            _human_step(6, "🧠 Step 6: Generating Final Answer", f"AI summarizing {query_res.get('row_count', 0)} rows")
            t_summary_start = time.time()
            summary_res = mcp_summarize_results(
                question=encoded_question,
                columns=query_res["columns"],
                rows=query_res["rows"],
                row_count=query_res["row_count"],
                trace_id=trace_id,
            )
            summary_llm_time = time.time() - t_summary_start
            masked_summary = summary_res["summary"]
            _human_log(f"   -> Summary LLM response time: {summary_llm_time:.2f}s")
            yield _emit_done(6, masked_summary)

            yield _emit_start(7)
            _human_step(7, "🔓 Step 7: Restoring Sensitive Data for Final Output")
            final_answer = mcp_pii_decode(masked_summary, trace_id=trace_id)["decoded_text"]
            yield _emit_done(7, final_answer)

            logger.info(f"[trace={trace_id}] FINAL_ANSWER_READY answer_len={len(str(final_answer))}")
            execution_time = time.time() - start_time
            logger.info(f"[trace={trace_id}] STREAM_DONE status=success execution_time={round(execution_time,2)}")
            emit_trace_event(
                event="QUERY_DONE",
                trace_id=trace_id,
                tool="agent_stream",
                payload={
                    "status": "success",
                    "execution_time_seconds": round(execution_time, 3),
                    "execution_time_ms": elapsed_ms(timer),
                    "steps": 7,
                },
            )
            _human_banner(
                "✅ Query Completed Successfully",
                [
                    ("Trace ID", trace_id),
                    ("Records", str(query_res.get("row_count", 0))),
                    ("SQL LLM", f"{sql_llm_time:.2f}s"),
                    ("DB Time", f"{db_exec_time:.2f}s"),
                    ("Summary LLM", f"{summary_llm_time:.2f}s"),
                    ("Exec Time", f"{round(execution_time, 2)} seconds"),
                ],
            )
            _human_banner(
                "🔐 Security Summary",
                [
                    ("PII Detected", str(pii_summary.get("count", 0))),
                    ("LLM Masked Input", "YES"),
                    ("SQL Restored for DB", "YES"),
                    ("Results Masked pre-LLM", "YES"),
                    ("Final Output Decoded", "YES"),
                ],
            )

            words = str(final_answer).split(" ")
            for idx, word in enumerate(words):
                yield {"type": "answer_chunk", "content": word + (" " if idx < len(words) - 1 else "")}

            done_event = {
                "type": "done",
                "execution_time": round(execution_time, 2),
                "reasoning_steps": 7,
                "trace_id": trace_id,
                "data": last_query_results_decoded or last_query_results,
                "pii": pii_summary,
            }
            logger.info(
                f"[trace={trace_id}] FRONTEND_DATA_SENT rows={query_res.get('row_count', '?') if isinstance(query_res, dict) else '?'}"
            )
            yield done_event
            return

        except Exception as e:
            execution_time = time.time() - start_time
            logger.exception(f"[trace={trace_id}] STREAM_DONE status=error execution_time={round(execution_time,2)} error={e}")
            emit_trace_event(
                event="QUERY_DONE",
                trace_id=trace_id,
                tool="agent_stream",
                payload={
                    "status": "error",
                    "error": str(e),
                    "execution_time_seconds": round(execution_time, 3),
                    "execution_time_ms": elapsed_ms(timer),
                },
                level="error",
            )
            yield {
                "type": "error",
                "error": str(e),
                "execution_time": round(execution_time, 2),
                "reasoning_steps": 0,
                "trace_id": trace_id,
            }


# Global agent instance (initialized on first use)
_agent_instance = None


def get_agent() -> MySQLAnalyticalAgent:
    """
    Get or create the global agent instance.

    Returns:
        MySQLAnalyticalAgent instance
    """
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = MySQLAnalyticalAgent()
    return _agent_instance


if __name__ == "__main__":
    # Test the agent
    agent = get_agent()

    test_question = "My name is Jyothika and I live in Narasapur. How many orders have I placed?"
    result = agent.query(test_question)

    print("\n" + "="*80)
    print("ANSWER:")
    print(result["answer"])
    print(f"\nExecution Time: {result['execution_time']}s")
    print(f"Reasoning Steps: {result['reasoning_steps']}")
    print("="*80)
