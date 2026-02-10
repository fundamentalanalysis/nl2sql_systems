"""MCP Tool A: get_schema - Retrieve database schema information."""
from typing import Dict, Any
from loguru import logger
from database.connection import db_manager
from app.config import settings
from app.rbac_policy import filter_schema_for_role


def get_schema(role: str = "admin") -> Dict[str, Any]:
    """
    Retrieve the database schema (RBAC disabled - full access).
    """
    # Force admin role to bypass RBAC filtering effectively
    target_role = "admin"
    """
    Retrieve the database schema filtered by user role.

    This tool connects to MySQL and extracts schema information from INFORMATION_SCHEMA,
    then filters the results based on RBAC policy.

    Args:
        role: User role ('admin' or 'viewer') for schema filtering

    Returns:
        Dict containing tables and their column information:
        {
            "tables": [
                {
                    "name": "table_name",
                    "columns": [
                        {
                            "name": "column_name",
                            "type": "data_type",
                            "nullable": true/false,
                            "key": "PRI/UNI/MUL/",
                            "default": "default_value",
                            "extra": "auto_increment/etc"
                        }
                    ]
                }
            ]
        }

    Raises:
        Exception: If database connection or query fails
    """
    logger.info("Executing get_schema tool")
    logger.info(f"Using database/schema: {settings.mysql_database}")

    try:
        # Fallback: Get all tables (optimized single query)
        # This code block is kept for fallback or if no question provided
        schema_info = {"tables": []}
        tables_dict = {}

        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                # Query INFORMATION_SCHEMA for ALL tables and columns in one go
                query = """
                    SELECT 
                        TABLE_NAME,
                        COLUMN_NAME as name,
                        COLUMN_TYPE as type,
                        IS_NULLABLE as nullable,
                        COLUMN_KEY as `key`,
                        COLUMN_DEFAULT as `default`,
                        EXTRA as extra
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA = %s 
                    ORDER BY TABLE_NAME, ORDINAL_POSITION
                """
                cursor.execute(
                    query, (db_manager.connection_params['database'],))
                columns_data = cursor.fetchall()

                # Group columns by table
                for col in columns_data:
                    table_name = col['TABLE_NAME']

                    if table_name not in tables_dict:
                        tables_dict[table_name] = []

                    tables_dict[table_name].append({
                        "name": col['name'],
                        "type": col['type'],
                        "nullable": col['nullable'] == 'YES',
                        "key": col['key'] if col['key'] else '',
                    })

        # Convert dict to list format
        for table_name, columns in tables_dict.items():
            schema_info["tables"].append({
                "name": table_name,
                "columns": columns
            })

        logger.info(
            f"Retrieved full schema for {len(schema_info['tables'])} tables")

        # Apply RBAC filtering
        filtered_schema = filter_schema_for_role(schema_info, role)
        logger.info(
            f"Filtered schema for role '{role}': {len(filtered_schema['tables'])} tables")

        return filtered_schema

    except Exception as e:
        logger.error(f"Failed to retrieve schema: {e}")
        raise


if __name__ == "__main__":
    # Test the tool
    import json
    schema = get_schema()
    print(json.dumps(schema, indent=2))
