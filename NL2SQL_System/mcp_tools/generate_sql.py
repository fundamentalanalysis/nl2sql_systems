"""MCP Tool B: generate_sql - Convert natural language to SQL query."""
from typing import Dict, Any, Optional
from loguru import logger
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import SecretStr
from app.config import settings
import re
from utils.trace_logger import emit_trace_event, start_timer, elapsed_ms


def _clip(text: Any, limit: int = 300) -> str:
    s = str(text)
    return s if len(s) <= limit else s[:limit] + "...(truncated)"


def _extract_usage(response: Any) -> Dict[str, Any]:
    usage = getattr(response, "usage_metadata", None)
    if usage:
        return usage
    response_metadata = getattr(response, "response_metadata", None)
    if isinstance(response_metadata, dict):
        return response_metadata.get("token_usage", {}) or response_metadata.get("usage", {})
    return {}


def _serialize_messages(messages: Any) -> Any:
    serialized = []
    for msg in messages:
        serialized.append(
            {
                "type": getattr(msg, "type", None),
                "content": getattr(msg, "content", str(msg)),
                "additional_kwargs": getattr(msg, "additional_kwargs", {}),
            }
        )
    return serialized


def generate_sql(question: str, schema: Dict[str, Any], trace_id: Optional[str] = None) -> Dict[str, str]:
    """
    Convert a natural language question to a SQL query using LLM.

    This tool uses the database schema and LLM to generate safe, optimized SQL queries.
    It emphasizes generating only SELECT statements and avoiding harmful operations.

    Args:
        question: Natural language question to convert to SQL
        schema: Database schema information from get_schema tool

    Returns:
        Dict containing the generated SQL:
        {
            "sql": "SELECT ... FROM ... WHERE ..."
        }

    Raises:
        ValueError: If generated SQL contains harmful operations
        Exception: If LLM call fails
    """
    trace_id = trace_id or "no-trace"
    timer = start_timer()
    logger.info(f"Generating SQL for question: {question}")
    emit_trace_event(
        event="TOOL_INPUT",
        trace_id=trace_id,
        tool="generate_sql_tool",
        payload={
            "question": question,
            "schema": schema,
        },
    )

    try:
        # Initialize LLM
        llm = AzureChatOpenAI(
            azure_endpoint=settings.azure_ai_endpoint,
            api_key=SecretStr(settings.azure_ai_api_key),
            model=settings.azure_openai_deployment,
            api_version=settings.azure_ai_api_version,
            temperature=0,  # Deterministic output for SQL generation
        )

        # Format schema for prompt
        schema_text = _format_schema_for_prompt(schema)
        emit_trace_event(
            event="TRANSFORM",
            trace_id=trace_id,
            tool="generate_sql_tool",
            payload={
                "name": "schema_formatted_for_prompt",
                "schema_text": schema_text,
            },
        )

        # Create prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert SQL query generator. Your task is to convert natural language questions into valid MySQL SELECT queries.

CRITICAL RULES:
1. ONLY generate SELECT queries - NEVER use UPDATE, DELETE, INSERT, ALTER, DROP, or TRUNCATE
2. Use the provided schema information to ensure accurate table and column names
3. Use proper JOINs when multiple tables are needed
4. Use appropriate WHERE clauses for filtering
5. Use GROUP BY and aggregate functions when analyzing data
6. Use ORDER BY to sort results logically
7. Always use proper MySQL syntax
8. Return ONLY the SQL query without explanations or markdown formatting
9. Do not include semicolons at the end
10. Use table aliases to improve readability when joining tables
11. For SUM/AVG aggregations, wrap with COALESCE(..., 0) to avoid NULL results when no rows match
12. NEVER use SELECT *; always select only the minimum required columns
13. If user asks for "products" or product list only, return product_name (and product_id only if explicitly requested)
14. For questions using wording like "supplied by", "supplier", or "vendor", JOIN products to suppliers on supplier_id and filter by suppliers.supplier_name
15. Use products.brand only for brand/manufacturer questions; do not treat supplier_name as brand
16. For item-status queries (e.g., cancelled/shipped/pending items), default to human-readable item details (product_name + item_status)
17. Include ID columns (order_item_id, product_id, order_id) only when the user explicitly asks for IDs
18. For person-name lookups, use STRICT exact matching only (case-insensitive), not fuzzy matching
19. If schema has first_name and last_name and the person value is a single string/token, match full name exactly with LOWER(CONCAT_WS(' ', first_name, last_name)) = LOWER('<person_value>')
20. If first_name and last_name are clearly separate values, match both exactly with LOWER(first_name)=LOWER(...) AND LOWER(last_name)=LOWER(...)
21. Do NOT use LIKE, SOUNDEX, or phonetic/similarity matching for names unless the user explicitly asks for similar matches
22. For location filters, normalize common suffix words in user text literals before comparison (e.g., Country, City, State) so 'Cuba Country' matches stored 'Cuba'
23. Prefer normalized comparison for locations: LOWER(column) = LOWER(TRIM(REPLACE(REPLACE(REPLACE(value, ' Country', ''), ' City', ''), ' State', '')))
24. For list-style postal code queries, prefer DISTINCT to avoid duplicate rows unless user explicitly asks for all rows including duplicates
25. For customer-list queries, default to only customer names (first_name, last_name)
26. Include customer email/phone/address only if the user explicitly asks for those fields
27. If user asks for customer/person "details", "full details", "complete details", or "profile", return comprehensive customer fields: customer_id, first_name, last_name, email, phone, country, state, city, postal_code, customer_segment, registration_date, last_purchase_date, is_active
28. Rule 27 overrides rule 25 when both could apply

Database Schema:
{schema}

Example questions and their SQL:
- "How many users are there?" -> SELECT COUNT(*) as user_count FROM users
- "Show top 10 products by price" -> SELECT product_name, price FROM products ORDER BY price DESC LIMIT 10
- "What is the average order value?" -> SELECT AVG(total) as avg_order_value FROM orders
- "I want to know the emails of Angel Hill" -> SELECT email FROM customers WHERE LOWER(first_name) = LOWER('Angel') AND LOWER(last_name) = LOWER('Hill')
- "I want to know the emails of [PERSON_AB12CD34]" -> SELECT email FROM customers WHERE LOWER(CONCAT_WS(' ', first_name, last_name)) = LOWER('[PERSON_AB12CD34]')
- "Show products supplied by Frazier Ltd" -> SELECT p.product_name FROM products p JOIN suppliers s ON p.supplier_id = s.supplier_id WHERE LOWER(s.supplier_name) = LOWER('Frazier Ltd')
- "Find all cancelled items" -> SELECT p.product_name, oi.item_status FROM order_items oi JOIN products p ON oi.product_id = p.product_id WHERE LOWER(oi.item_status) = LOWER('Cancelled')
- "Find all cancelled item IDs" -> SELECT oi.order_item_id, oi.product_id, oi.item_status FROM order_items oi WHERE LOWER(oi.item_status) = LOWER('Cancelled')
- "What are the postal codes of customers from [LOCATION_XXXX] Country" -> SELECT postal_code FROM customers WHERE LOWER(country) = LOWER(TRIM(REPLACE('[LOCATION_XXXX] Country', ' Country', '')))
- "What are the postal codes of customers from [LOCATION_XXXX] Country (unique list)" -> SELECT DISTINCT postal_code FROM customers WHERE LOWER(country) = LOWER(TRIM(REPLACE('[LOCATION_XXXX] Country', ' Country', '')))
- "Show customers from Texas" -> SELECT first_name, last_name FROM customers WHERE LOWER(state) = LOWER('Texas')
- "Give me the details of [PERSON_AB12CD34]" -> SELECT customer_id, first_name, last_name, email, phone, country, state, city, postal_code, customer_segment, registration_date, last_purchase_date, is_active FROM customers WHERE LOWER(CONCAT_WS(' ', first_name, last_name)) = LOWER('[PERSON_AB12CD34]')
"""),
            ("human", "{question}")
        ])

        rendered_messages = prompt_template.format_messages(
            schema=schema_text,
            question=question,
        )
        emit_trace_event(
            event="LLM_PROMPT",
            trace_id=trace_id,
            tool="generate_sql_tool",
            payload={
                "messages": _serialize_messages(rendered_messages),
            },
        )

        # Generate SQL (audit: log the exact masked question sent to LLM)
        logger.info(f"LLM_INVOKE tool=generate_sql_tool masked_question={_clip(question)}")
        chain = prompt_template | llm
        response = chain.invoke({
            "schema": schema_text,
            "question": question
        })
        logger.info(f"LLM_RESPONSE tool=generate_sql_tool raw_response={_clip(response.content)}")
        emit_trace_event(
            event="LLM_RAW_RESPONSE",
            trace_id=trace_id,
            tool="generate_sql_tool",
            payload={
                "response_content": response.content,
                "response_metadata": getattr(response, "response_metadata", {}),
                "usage": _extract_usage(response),
            },
        )

        parsed_sql = response.content.strip()
        emit_trace_event(
            event="PARSED_OUTPUT",
            trace_id=trace_id,
            tool="generate_sql_tool",
            payload={"parsed_sql_before_cleanup": parsed_sql},
        )

        # Clean up the SQL
        sql = parsed_sql.replace("```sql", "").replace("```", "").strip()
        emit_trace_event(
            event="TRANSFORM",
            trace_id=trace_id,
            tool="generate_sql_tool",
            payload={
                "name": "sql_cleanup",
                "input_sql": parsed_sql,
                "output_sql": sql,
            },
        )

        # Validate the SQL is safe
        _validate_sql_safety(sql)

        logger.info(f"Generated SQL: {sql}")
        result = {"sql": sql}
        emit_trace_event(
            event="TOOL_OUTPUT",
            trace_id=trace_id,
            tool="generate_sql_tool",
            payload={
                "result": result,
                "execution_time_ms": elapsed_ms(timer),
            },
        )
        return result

    except Exception as e:
        logger.error(f"Failed to generate SQL: {e}")
        emit_trace_event(
            event="TOOL_ERROR",
            trace_id=trace_id,
            tool="generate_sql_tool",
            payload={
                "error": str(e),
                "execution_time_ms": elapsed_ms(timer),
            },
            level="error",
        )
        raise


def _format_schema_for_prompt(schema: Dict[str, Any]) -> str:
    """
    Format schema information into a readable string for the LLM prompt.

    Args:
        schema: Schema dict from get_schema

    Returns:
        Formatted schema string
    """
    lines = []
    for table in schema.get("tables", []):
        lines.append(f"\nTable: {table['name']}")
        lines.append("Columns:")
        for col in table.get("columns", []):
            key_info = f" [{col['key']}]" if col['key'] else ""
            nullable = "NULL" if col['nullable'] else "NOT NULL"
            lines.append(
                f"  - {col['name']}: {col['type']} {nullable}{key_info}")

    return "\n".join(lines)


def _validate_sql_safety(sql: str) -> None:
    """
    Validate that SQL query is safe (SELECT only).

    Args:
        sql: SQL query to validate

    Raises:
        ValueError: If SQL contains harmful operations
    """
    sql_upper = sql.upper()

    # List of forbidden SQL keywords
    forbidden_keywords = [
        'UPDATE', 'DELETE', 'INSERT', 'ALTER',
        'DROP', 'TRUNCATE', 'CREATE',
        'GRANT', 'REVOKE', 'EXECUTE'
    ]

    for keyword in forbidden_keywords:
        # Use word boundaries to avoid false positives
        if re.search(r'\b' + keyword + r'\b', sql_upper):
            raise ValueError(
                f"Harmful SQL operation detected: {keyword}. Only SELECT queries are allowed.")

    # Allow REPLACE(...) function in SELECT expressions, but block REPLACE statements.
    if re.match(r'^\s*REPLACE\b', sql_upper):
        raise ValueError(
            "Harmful SQL operation detected: REPLACE statement. Only SELECT queries are allowed.")

    # Ensure it's a SELECT query
    if not sql_upper.strip().startswith('SELECT'):
        raise ValueError("Query must be a SELECT statement")

    logger.debug("SQL safety validation passed")
