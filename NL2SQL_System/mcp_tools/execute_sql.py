"""MCP Tool C: execute_sql - Safely execute SQL queries with RBAC enforcement."""
from typing import Dict, List, Any, Optional
from loguru import logger
from database.connection import db_manager
import re
from app.rbac_policy import is_authorized
import sqlparse
from utils.trace_logger import emit_trace_event, start_timer, elapsed_ms


def execute_sql(sql: str, role: str = "admin", trace_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Safely execute a SQL query with automatic safety checks, LIMIT enforcement, and RBAC validation.

    This tool validates and executes SQL queries with the following safety features:
    - Rejects UPDATE, DELETE, INSERT, ALTER, DROP, and TRUNCATE operations
    - Automatically applies LIMIT 200 if not present
    - Validates table and column access based on user role
    - Returns structured results with column names and row data

    Args:
        sql: SQL query to execute (must be SELECT only)
        role: User role ('admin' or 'viewer') for access control

    Returns:
        Dict containing query results:
        {
            "columns": ["col1", "col2", ...],
            "rows": [[val1, val2, ...], ...],
            "row_count": 123
        }

    Raises:
        ValueError: If SQL contains harmful operations or unauthorized access
        Exception: If query execution fails
    """
    trace_id = trace_id or "no-trace"
    timer = start_timer()
    logger.info(f"Executing SQL: {sql[:100]}...")
    emit_trace_event(
        event="TOOL_INPUT",
        trace_id=trace_id,
        tool="execute_sql_tool",
        payload={
            "sql": sql,
            "role": role,
        },
    )

    try:
        # Validate SQL safety
        _validate_sql_safety(sql)
        emit_trace_event(
            event="INTERNAL_ACTION",
            trace_id=trace_id,
            tool="execute_sql_tool",
            payload={"action": "sql_safety_validated"},
        )

        # Apply RBAC validation
        _validate_sql_access(sql, role)
        emit_trace_event(
            event="INTERNAL_ACTION",
            trace_id=trace_id,
            tool="execute_sql_tool",
            payload={"action": "rbac_validated", "role": role},
        )

        # 1. Restore tokens to original values for DB execution
        from privacy.decoder import decode_text
        sql_to_execute = decode_text(sql)
        emit_trace_event(
            event="TRANSFORM",
            trace_id=trace_id,
            tool="execute_sql_tool",
            payload={
                "name": "decode_sql_tokens",
                "input_sql": sql,
                "output_sql": sql_to_execute,
            },
        )
        if sql_to_execute != sql:
            logger.info("Restored PII tokens in SQL query")

        # Apply LIMIT if not present
        sql_with_limit = _apply_limit(sql_to_execute)
        emit_trace_event(
            event="TRANSFORM",
            trace_id=trace_id,
            tool="execute_sql_tool",
            payload={
                "name": "apply_limit",
                "input_sql": sql_to_execute,
                "output_sql": sql_with_limit,
            },
        )

        # Execute query
        emit_trace_event(
            event="DB_QUERY",
            trace_id=trace_id,
            tool="execute_sql_tool",
            payload={"sql": sql_with_limit},
        )
        columns, rows, row_count = db_manager.execute_query(sql_with_limit)
        emit_trace_event(
            event="DB_RAW_RESULT",
            trace_id=trace_id,
            tool="execute_sql_tool",
            payload={
                "columns": columns,
                "rows": rows,
                "row_count": row_count,
            },
        )

        # 2. Re-encode database results (mask real PII with tokens)
        from privacy.encoder import encode_results
        encoded_rows = encode_results(columns, rows)
        emit_trace_event(
            event="TRANSFORM",
            trace_id=trace_id,
            tool="execute_sql_tool",
            payload={
                "name": "encode_result_pii",
                "input_rows": rows,
                "output_rows": encoded_rows,
            },
        )
        if encoded_rows != rows:
            logger.info("Masked PII in database results")

        result = {
            "columns": columns,
            "rows": encoded_rows,
            "row_count": row_count
        }

        logger.info(
            f"Query executed successfully for role '{role}', returned {row_count} rows (masked)")
        emit_trace_event(
            event="TOOL_OUTPUT",
            trace_id=trace_id,
            tool="execute_sql_tool",
            payload={
                "result": result,
                "execution_time_ms": elapsed_ms(timer),
            },
        )
        return result

    except Exception as e:
        logger.error(f"Failed to execute SQL: {e}")
        emit_trace_event(
            event="TOOL_ERROR",
            trace_id=trace_id,
            tool="execute_sql_tool",
            payload={
                "error": str(e),
                "execution_time_ms": elapsed_ms(timer),
            },
            level="error",
        )
        raise


def _validate_sql_safety(sql: str) -> None:
    """
    Validate that SQL query is safe to execute.

    Args:
        sql: SQL query to validate

    Raises:
        ValueError: If SQL contains harmful operations
    """
    sql_upper = sql.upper().strip()

    # List of forbidden SQL keywords
    forbidden_keywords = [
        'UPDATE', 'DELETE', 'INSERT', 'ALTER',
        'DROP', 'TRUNCATE', 'CREATE',
        'GRANT', 'REVOKE', 'EXECUTE', 'CALL',
        'LOAD', 'OUTFILE', 'INFILE'
    ]

    for keyword in forbidden_keywords:
        # Use word boundaries to avoid false positives
        if re.search(r'\b' + keyword + r'\b', sql_upper):
            raise ValueError(
                f"Harmful SQL operation detected: {keyword}. "
                "Only SELECT queries are allowed for security reasons."
            )

    # Allow REPLACE(...) function in SELECT expressions, but block REPLACE statements.
    if re.match(r'^\s*REPLACE\b', sql_upper):
        raise ValueError(
            "Harmful SQL operation detected: REPLACE statement. "
            "Only SELECT queries are allowed for security reasons."
        )

    # Ensure it starts with SELECT
    if not sql_upper.startswith('SELECT'):
        raise ValueError("Query must be a SELECT statement")

    # Check for multiple statements (SQL injection attempt)
    if ';' in sql.rstrip(';'):
        raise ValueError("Multiple SQL statements are not allowed")

    logger.debug("SQL safety validation passed")


def _validate_sql_access(sql: str, role: str) -> None:
    """
    Validate that SQL query accesses only authorized tables and columns.
    Uses sqlglot for table extraction and sqlparse for column extraction.

    Args:
        sql: SQL query to validate
        role: User role for access control

    Raises:
        ValueError: If query references unauthorized tables or columns
    """
    # Import both libraries
    try:
        import sqlglot
        from sqlglot import exp
    except ImportError:
        logger.error("sqlglot not installed")
        raise ImportError("sqlglot is required for RBAC validation")

    try:
        from sqlparse.sql import IdentifierList, Identifier
        from sqlparse.tokens import Keyword, DML
    except ImportError:
        logger.error("sqlparse not installed")
        raise ImportError(
            "sqlparse is required for column-level RBAC validation")

    try:
        # Parse SQL into AST with sqlglot for table extraction
        parsed = sqlglot.parse_one(sql)
    except Exception as e:
        raise ValueError(f"Failed to parse SQL query with sqlglot: {e}")

    # Extract tables using sqlglot
    tables_referenced = set()
    for table_exp in parsed.find_all(exp.Table):
        table_name = table_exp.name
        # Handle quoted table names
        if hasattr(table_exp, 'this') and hasattr(table_exp.this, 'this'):
            table_name = table_exp.this.this
        tables_referenced.add(str(table_name))

    logger.debug(f"Extracted tables for validation: {tables_referenced}")

    # Validate table access
    for table in tables_referenced:
        # Ignore empty table names if any
        if not table:
            continue

        if not is_authorized(role, table):
            raise ValueError(
                f"Access denied: Table '{table}' not authorized for role '{role}'. "
                "Contact administrator for access."
            )

    # Extract columns using sqlparse
    columns_referenced = _extract_columns_with_sqlparse(sql, tables_referenced)
    logger.debug(f"Extracted columns for validation: {columns_referenced}")

    # Validate column access
    for table, column in columns_referenced:
        if not is_authorized(role, table, column):
            raise ValueError(
                f"Access denied: Column '{table}.{column}' not authorized for role '{role}'. "
                "Contact administrator for access."
            )

    logger.debug(
        f"RBAC validation passed for role '{role}' on tables: {tables_referenced}, columns: {columns_referenced}")


def _extract_columns_with_sqlparse(sql: str, tables_referenced: set) -> list:
    """
    Extract column references from SQL using sqlparse.

    Args:
        sql: SQL query string
        tables_referenced: Set of tables identified by sqlglot

    Returns:
        List of tuples (table_name, column_name) representing column references
    """
    parsed = sqlparse.parse(sql)[0]
    columns_referenced = []

    # Look for identifiers in the parsed tokens
    for token in parsed.flatten():
        if token.ttype is not None and '.' in str(token):
            # This looks like a table.column reference
            parts = str(token).split('.')
            if len(parts) == 2:
                table_part = parts[0].strip('"`[]')  # Remove quotes if any
                column_part = parts[1].strip('"`[]')  # Remove quotes if any

                # Validate that the table part is actually a known table
                if table_part in tables_referenced:
                    columns_referenced.append((table_part, column_part))

    # Also check for column references in SELECT, WHERE, etc. clauses
    def extract_identifiers_from_token(token):
        identifiers = []
        if hasattr(token, 'tokens'):
            for sub_token in token.tokens:
                identifiers.extend(extract_identifiers_from_token(sub_token))
        else:
            # Check for table.column pattern
            token_str = str(token).strip()
            if '.' in token_str and not token_str.upper() in ['SELECT', 'FROM', 'WHERE', 'ORDER', 'GROUP', 'HAVING']:
                parts = token_str.split('.')
                if len(parts) == 2:
                    table_part = parts[0].strip('"`[]')
                    column_part = parts[1].strip('"`[]')

                    # Validate that the table part is actually a known table
                    if table_part in tables_referenced:
                        identifiers.append((table_part, column_part))
        return identifiers

    # Process the entire parsed query
    identifiers = extract_identifiers_from_token(parsed)
    columns_referenced.extend(identifiers)

    # Remove duplicates
    return list(set(columns_referenced))


def _apply_limit(sql: str) -> str:
    """
    Apply LIMIT 200 to SQL query if not already present.

    Args:
        sql: Original SQL query

    Returns:
        SQL query with LIMIT applied
    """
    sql_upper = sql.upper()

    # Check if LIMIT already exists
    if 'LIMIT' in sql_upper:
        logger.debug("LIMIT already present in query")
        return sql

    # Add LIMIT 200
    sql_with_limit = sql.rstrip(';').strip() + ' LIMIT 200'
    logger.debug("Applied automatic LIMIT 200 to query")

    return sql_with_limit
