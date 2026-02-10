"""Database connection management for MySQL."""
from contextlib import contextmanager
from typing import Generator, Any, List, Tuple
import pymysql
from pymysql.cursors import DictCursor
from loguru import logger
from app.config import settings
from decimal import Decimal
from datetime import datetime, date, time, timedelta


def convert_to_json_serializable(obj: Any) -> Any:
    """
    Convert MySQL types to JSON-serializable Python types.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, time):
        return obj.isoformat()
    elif isinstance(obj, timedelta):
        return str(obj)
    elif isinstance(obj, bytes):
        return obj.decode('utf-8', errors='ignore')
    return obj


class MySQLConnection:
    """MySQL connection manager with connection pooling support."""
    
    def __init__(self):
        """Initialize the connection manager."""
        self.connection_params = {
            'host': settings.mysql_host,
            'port': settings.mysql_port,
            'user': settings.mysql_user,
            'password': settings.mysql_password,
            'database': settings.mysql_database,
            'charset': 'utf8mb4',
            'cursorclass': DictCursor,
        }
        logger.info(f"MySQL connection manager initialized for {settings.mysql_host}:{settings.mysql_port}/{settings.mysql_database}")
    
    @contextmanager
    def get_connection(self) -> Generator[pymysql.connections.Connection, None, None]:
        """
        Context manager for MySQL connections.
        
        Yields:
            pymysql.connections.Connection: Active MySQL connection
            
        Example:
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM users")
        """
        connection = None
        try:
            connection = pymysql.connect(**self.connection_params)
            logger.debug("MySQL connection established")
            yield connection
            connection.commit()
        except pymysql.Error as e:
            if connection:
                connection.rollback()
            logger.error(f"MySQL error: {e}")
            raise
        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"Unexpected error: {e}")
            raise
        finally:
            if connection:
                connection.close()
                logger.debug("MySQL connection closed")
    
    def execute_query(self, query: str) -> Tuple[List[str], List[List[Any]], int]:
        """
        Execute a SQL query and return results.
        
        Args:
            query: SQL query to execute
            
        Returns:
            Tuple of (column_names, rows, row_count)
            
        Raises:
            pymysql.Error: If query execution fails
        """
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                logger.info(f"Executing query: {query}...")
                cursor.execute(query)
                
                # Get column names
                if cursor.description:
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()
                    
                    # Convert dict rows to list of lists and make JSON-serializable
                    row_data = [
                        [convert_to_json_serializable(row[col]) for col in columns] 
                        for row in rows
                    ]
                    row_count = len(row_data)
                    
                    logger.info(f"Query returned {row_count} rows")
                    return columns, row_data, row_count
                else:
                    # No results (e.g., INSERT, UPDATE)
                    return [], [], 0
    
    def test_connection(self) -> bool:
        """
        Test the database connection.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        if getattr(settings, "database_mode", "auto").lower() == "csv":
            logger.info("Database mode is 'csv'; skipping MySQL connection test")
            return True

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
                    logger.info("Database connection test successful")
                    return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def get_all_tables(self) -> List[str]:
        """
        Get list of all tables in the database.
        
        Returns:
            List of table names
        """
        query = """
            SELECT TABLE_NAME 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_SCHEMA = %s
            ORDER BY TABLE_NAME
        """
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (settings.mysql_database,))
                results = cursor.fetchall()
                tables = [row['TABLE_NAME'] for row in results]
                logger.info(f"Found {len(tables)} tables in database")
                return tables


# Global connection manager instance
db_manager = MySQLConnection()
