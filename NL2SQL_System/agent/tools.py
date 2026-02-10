"""LangChain tool wrappers for MCP tools."""
from mcp_tools.summarize_results import summarize_results as mcp_summarize_results
from mcp_tools.execute_sql import execute_sql as mcp_execute_sql
from mcp_tools.generate_sql import generate_sql as mcp_generate_sql
from mcp_tools.get_schema import get_schema as mcp_get_schema
from mcp_tools.pii_detect import pii_detect as mcp_pii_detect
from mcp_tools.pii_encode import pii_encode as mcp_pii_encode
from mcp_tools.pii_decode import pii_decode as mcp_pii_decode
from typing import Dict, Any, List
from langchain.tools import tool
from loguru import logger

# Import MCP tools
import sys
sys.path.append('.')


@tool
def get_schema_tool(role: str = "admin") -> Dict[str, Any]:
    """
    Retrieve the database schema filtered by user role.

    Args:
        role: User role ('admin' or 'viewer') for schema filtering

    Returns:
        Dict containing tables and their column information.
    """
    return mcp_get_schema(role)


@tool
def generate_sql_tool(question: str, db_schema: str) -> str:
    """
    Convert a natural language question into a SQL query.

    This tool uses the database schema and LLM to generate safe, optimized SQL SELECT statements.
    It prevents generation of harmful SQL operations (UPDATE, DELETE, INSERT, etc.).

    Args:
        question: Natural language question to convert to SQL
        db_schema: Database schema information as JSON string from get_schema_tool

    Returns:
        Generated SQL query string
    """
    logger.info("LangChain tool: generate_sql_tool invoked")
    import json

    # Parse schema if it's a string
    if isinstance(db_schema, str):
        schema_dict = json.loads(db_schema)
    else:
        schema_dict = db_schema

    result = mcp_generate_sql(question, schema_dict)
    return result["sql"]


@tool
def execute_sql_tool(sql: str, role: str = "admin") -> str:
    """
    Safely execute a SQL query with automatic safety checks, LIMIT enforcement, and RBAC validation.

    This tool validates and executes SQL queries with safety features:
    - Rejects UPDATE, DELETE, INSERT, ALTER, DROP, and TRUNCATE operations
    - Automatically applies LIMIT 200 if not present
    - Validates table and column access based on user role
    - Returns structured results with column names and row data

    Args:
        sql: SQL query to execute (must be SELECT only)
        role: User role ('admin' or 'viewer') for access control

    Returns:
        JSON string containing query results with columns, rows, and row_count
    """
    logger.info("LangChain tool: execute_sql_tool invoked")
    import json

    result = mcp_execute_sql(sql, role)
    return json.dumps(result)


@tool
def summarize_results_tool(question: str, results: str) -> str:
    """
    Generate an intelligent, human-readable summary of query results.

    This tool analyzes query results and provides insights, patterns, trends, and business implications.
    Use this tool LAST to create the final answer for the user.

    Args:
        question: Original natural language question from the user
        results: Query results as JSON string from execute_sql_tool

    Returns:
        Natural language summary with insights
    """
    logger.info("LangChain tool: summarize_results_tool invoked")
    import json

    # Parse results
    results_dict = json.loads(results)

    summary_result = mcp_summarize_results(
        question=question,
        columns=results_dict["columns"],
        rows=results_dict["rows"],
        row_count=results_dict["row_count"]
    )

    return summary_result["summary"]


@tool
def pii_detect_tool(text: str) -> Dict[str, Any]:
    """
    Detect sensitive information (PII) like names, phone numbers, and locations in text.
    Uses Presidio for high accuracy detection.

    Args:
        text: The text to analyze for PII

    Returns:
        Dict containing detected entities and their metadata.
    """
    return mcp_pii_detect(text)


@tool
def pii_encode_tool(text: str) -> Dict[str, Any]:
    """
    Replace sensitive PII in text with secure SHA-256 tokens.
    Original values are encrypted and stored securely in memory for later restoration.

    Args:
        text: The text to encode

    Returns:
        Dict containing the encoded_text and mapping metadata.
    """
    return mcp_pii_encode(text)


@tool
def pii_decode_tool(text: str) -> str:
    """
    Restore secure placeholder tokens in text back to their original sensitive values.
    Use this tool last before responding to the user if the text contains tokens.

    Args:
        text: Text containing tokens like [PERSON_XXXX]

    Returns:
        The decoded string with original values restored.
    """
    result = mcp_pii_decode(text)
    return result["decoded_text"]


# Export all tools
LANGCHAIN_TOOLS = [
    pii_detect_tool,
    pii_encode_tool,
    get_schema_tool,
    generate_sql_tool,
    execute_sql_tool,
    summarize_results_tool,
    pii_decode_tool
]
