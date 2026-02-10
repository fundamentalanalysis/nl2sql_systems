"""MCP Tool B: generate_sql - Convert natural language to SQL query."""
from typing import Dict, Any
from loguru import logger
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import SecretStr
from app.config import settings
import json
import re


def generate_sql(question: str, schema: Dict[str, Any]) -> Dict[str, str]:
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
    logger.info(f"Generating SQL for question: {question}")

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

Database Schema:
{schema}

Example questions and their SQL:
- "How many users are there?" -> SELECT COUNT(*) as user_count FROM users
- "Show top 10 products by price" -> SELECT * FROM products ORDER BY price DESC LIMIT 10
- "What is the average order value?" -> SELECT AVG(total) as avg_order_value FROM orders
"""),
            ("human", "{question}")
        ])

        # Generate SQL
        chain = prompt_template | llm
        response = chain.invoke({
            "schema": schema_text,
            "question": question
        })

        sql = response.content.strip()

        # Clean up the SQL
        sql = sql.replace("```sql", "").replace("```", "").strip()

        # Validate the SQL is safe
        _validate_sql_safety(sql)

        logger.info(f"Generated SQL: {sql}")
        return {"sql": sql}

    except Exception as e:
        logger.error(f"Failed to generate SQL: {e}")
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
        'DROP', 'TRUNCATE', 'CREATE', 'REPLACE',
        'GRANT', 'REVOKE', 'EXECUTE'
    ]

    for keyword in forbidden_keywords:
        # Use word boundaries to avoid false positives
        if re.search(r'\b' + keyword + r'\b', sql_upper):
            raise ValueError(
                f"Harmful SQL operation detected: {keyword}. Only SELECT queries are allowed.")

    # Ensure it's a SELECT query
    if not sql_upper.strip().startswith('SELECT'):
        raise ValueError("Query must be a SELECT statement")

    logger.debug("SQL safety validation passed")


if __name__ == "__main__":
    # Test the tool
    test_schema = {
        "tables": [
            {
                "name": "users",
                "columns": [
                    {"name": "id", "type": "int", "nullable": False, "key": "PRI"},
                    {"name": "email",
                        "type": "varchar(255)", "nullable": False, "key": ""}
                ]
            }
        ]
    }

    result = generate_sql("How many users are there?", test_schema)
    print(result)
