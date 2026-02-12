"""MCP Tool A: get_schema - Retrieve database schema information."""
from typing import Dict, Any, Optional
from loguru import logger
from database.connection import db_manager
from app.config import settings
from app.rbac_policy import filter_schema_for_role
from utils.trace_logger import emit_trace_event, start_timer, elapsed_ms


def get_schema(role: str = "admin", trace_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Retrieve the database schema with RBAC filtering disabled (full schema access).

    This tool connects to MySQL and extracts schema information from INFORMATION_SCHEMA,
    then returns the full schema by forcing admin-level visibility.

    Args:
        role: Requested user role (ignored for filtering; kept for compatibility)

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
    trace_id = trace_id or "no-trace"
    effective_role = "admin"
    timer = start_timer()
    logger.info("Executing get_schema tool")
    logger.info(f"Using database/schema: {settings.mysql_database}")
    emit_trace_event(
        event="TOOL_INPUT",
        trace_id=trace_id,
        tool="get_schema_tool",
        payload={
            "requested_role": role,
            "effective_role": effective_role,
            "database": settings.mysql_database,
        },
    )

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
        emit_trace_event(
            event="INTERNAL_ACTION",
            trace_id=trace_id,
            tool="get_schema_tool",
            payload={
                "action": "schema_loaded_from_information_schema",
                "table_count": len(schema_info["tables"]),
                "schema_info": schema_info,
            },
        )

        logger.info(
            f"Retrieved full schema for {len(schema_info['tables'])} tables")

        # RBAC intentionally bypassed for schema discovery by forcing admin visibility.
        filtered_schema = filter_schema_for_role(schema_info, effective_role)
        logger.info(
            f"Returned schema for effective role '{effective_role}' (requested: '{role}'): "
            f"{len(filtered_schema['tables'])} tables")
        emit_trace_event(
            event="TOOL_OUTPUT",
            trace_id=trace_id,
            tool="get_schema_tool",
            payload={
                "result": filtered_schema,
                "execution_time_ms": elapsed_ms(timer),
            },
        )

        return filtered_schema

    except Exception as e:
        logger.error(f"Failed to retrieve schema: {e}")
        emit_trace_event(
            event="TOOL_ERROR",
            trace_id=trace_id,
            tool="get_schema_tool",
            payload={
                "error": str(e),
                "execution_time_ms": elapsed_ms(timer),
            },
            level="error",
        )
        raise
