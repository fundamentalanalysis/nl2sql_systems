"""RBAC policy configuration - single source of truth for access control."""
from typing import Dict, List, Union

# RBAC Policy Definition
# Structure: role -> {tables: "*", columns: "*"} OR {tables: {table_name: [allowed_columns]}}
RBAC_POLICY: Dict[str, Dict[str, Union[str, Dict[str, List[str]]]]] = {
    "admin": {
        "tables": "*",
        "columns": "*",  # Full access to everything
    },
    "viewer": {
        "tables": {
            "customers": [
                "customer_id", "first_name", "last_name",
                "country", "state", "city",
                "customer_segment", "registration_date",
                "last_purchase_date", "is_active"  # Only these columns allowed
            ],
            "orders": [
                "order_id", "customer_id",
                "order_date", "delivery_date",
                "total_amount", "discount_applied",
                "tax_amount", "shipping_cost",
                "order_status", "payment_method"
            ],
            "order_items": [
                "order_item_id", "order_id", "product_id",
                "quantity", "unit_price", "line_total",
                "discount_percent", "tax_percent", "item_status"
            ],
            "products": [
                "product_id", "product_name",
                "category", "sub_category",
                "brand", "price", "stock_quantity", "description",
                "is_discontinued"
            ],
            "suppliers": [
                "supplier_id", "supplier_name", "is_active"
            ],
        }
    }
}


def is_authorized(role: str, table: str, column: str = None) -> bool:
    """
    Check if a role is authorized to access a table/column.

    Args:
        role: User role ('admin' or 'viewer')
        table: Table name to check
        column: Column name to check (optional)

    Returns:
        bool: True if authorized, False otherwise
    """
    if role not in RBAC_POLICY:
        return False

    policy = RBAC_POLICY[role]

    # Admin has full access
    if policy["tables"] == "*":
        return True

    # Check table access
    allowed_tables = policy["tables"]
    if table not in allowed_tables:
        return False

    # If no column specified, table access is sufficient
    if column is None:
        return True

    # Check column access
    allowed_columns = allowed_tables[table]
    return column in allowed_columns


def filter_schema_for_role(schema: dict, role: str) -> dict:
    """
    Filter database schema based on RBAC policy.

    Args:
        schema: Full schema dictionary from get_schema
        role: User role ('admin' or 'viewer')

    Returns:
        Filtered schema containing only allowed tables/columns
    """
    if RBAC_POLICY[role]["tables"] == "*":
        return schema

    allowed_tables = RBAC_POLICY[role]["tables"]
    filtered_schema = {"tables": []}

    for table_info in schema.get("tables", []):
        table_name = table_info["name"]

        if table_name in allowed_tables:
            allowed_columns = allowed_tables[table_name]
            filtered_columns = [
                col for col in table_info.get("columns", [])
                if col["name"] in allowed_columns
            ]

            filtered_schema["tables"].append({
                "name": table_name,
                "columns": filtered_columns
            })

    return filtered_schema
