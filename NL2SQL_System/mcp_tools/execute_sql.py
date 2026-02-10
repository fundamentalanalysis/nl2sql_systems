"""MCP Tool C: execute_sql - Safely execute SQL queries with RBAC enforcement."""
from typing import Dict, List, Any
from loguru import logger
from database.connection import db_manager
import re
from app.rbac_policy import is_authorized
import sqlparse


def execute_sql(sql: str, role: str = "admin") -> Dict[str, Any]:
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
    logger.info(f"Executing SQL: {sql[:100]}...")

    try:
        # Validate SQL safety
        _validate_sql_safety(sql)

        # Apply RBAC validation
        _validate_sql_access(sql, role)

        # 1. Restore tokens to original values for DB execution
        from privacy.decoder import decode_text
        sql_to_execute = decode_text(sql)
        if sql_to_execute != sql:
            logger.info("Restored PII tokens in SQL query")

        # Apply LIMIT if not present
        sql_with_limit = _apply_limit(sql_to_execute)

        # Execute query
        columns, rows, row_count = db_manager.execute_query(sql_with_limit)

        # 2. Re-encode database results (mask real PII with tokens)
        from privacy.encoder import encode_results
        encoded_rows = encode_results(columns, rows)
        if encoded_rows != rows:
            logger.info("Masked PII in database results")

        result = {
            "columns": columns,
            "rows": encoded_rows,
            "row_count": row_count
        }

        logger.info(
            f"Query executed successfully for role '{role}', returned {row_count} rows (masked)")
        return result

    except Exception as e:
        logger.error(f"Failed to execute SQL: {e}")
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
        'DROP', 'TRUNCATE', 'CREATE', 'REPLACE',
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


if __name__ == "__main__":
    # Test the tool
    import json

    # Test query without LIMIT
    result = execute_sql("SELECT * FROM users")
    print(json.dumps(result, indent=2))

    # Test query with LIMIT
    result = execute_sql("SELECT * FROM users LIMIT 10")
    print(json.dumps(result, indent=2))

    # Test harmful query (should fail)
    try:
        result = execute_sql("DELETE FROM users WHERE id = 1")
    except ValueError as e:
        print(f"Correctly rejected harmful query: {e}")
