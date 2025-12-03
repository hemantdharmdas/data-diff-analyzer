import pandas as pd
from sqlalchemy import create_engine, text
from typing import Dict, Optional, Tuple
import snowflake.connector

class DatabaseConnector:
    """Universal database connector for multiple database types."""
    
    SUPPORTED_DATABASES = ['mysql', 'postgresql', 'snowflake']
    
    def __init__(self):
        self.engine = None
        self.snowflake_conn = None
        self.db_config = None
    
    def create_connection_string(self, db_type: str, config: Dict) -> str:
        """
        Create SQLAlchemy connection string based on database type.
        
        Args:
            db_type: Type of database (mysql, postgresql)
            config: Dictionary with connection parameters
        
        Returns:
            SQLAlchemy connection string
        """
        db_type = db_type.lower()
        
        if db_type == 'mysql':
            return f"mysql+pymysql://{config['user']}:{config['password']}@{config['host']}:{config.get('port', 3306)}/{config['database']}"
        
        elif db_type == 'postgresql':
            return f"postgresql+psycopg2://{config['user']}:{config['password']}@{config['host']}:{config.get('port', 5432)}/{config['database']}"
        
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
    
    def connect(self, db_type: str, config: Dict) -> bool:
        """
        Establish database connection.
        
        Args:
            db_type: Type of database
            config: Connection configuration
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            db_type = db_type.lower()
            self.db_config = config
            
            if db_type == 'snowflake':
                connection_params = {
                    'user': config['user'],
                    'password': config['password'],
                    'account': config['account']
                }
                
                if config.get('warehouse'):
                    connection_params['warehouse'] = config['warehouse']
                if config.get('database'):
                    connection_params['database'] = config['database']
                if config.get('schema'):
                    connection_params['schema'] = config['schema']
                if config.get('role'):
                    connection_params['role'] = config['role']
                
                self.snowflake_conn = snowflake.connector.connect(**connection_params)
                
                cursor = self.snowflake_conn.cursor()
                if config.get('database'):
                    cursor.execute(f"USE DATABASE {config['database']}")
                if config.get('schema'):
                    cursor.execute(f"USE SCHEMA {config['schema']}")
                cursor.close()
                
                return True
            
            connection_string = self.create_connection_string(db_type, config)
            self.engine = create_engine(
                connection_string,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_pre_ping=True
            )
            
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            return True
        
        except Exception as e:
            raise ConnectionError(f"Failed to connect to {db_type}: {str(e)}")
    
    def read_table(self, table_name: str, schema: Optional[str] = None, 
                   limit: Optional[int] = None, where_clause: Optional[str] = None) -> pd.DataFrame:
        """
        Read data from a database table.
        
        Args:
            table_name: Name of the table
            schema: Schema name (optional)
            limit: Maximum number of rows to fetch
            where_clause: SQL WHERE clause for filtering (without WHERE keyword)
        
        Returns:
            Pandas DataFrame with table data
        """
        try:
            if schema:
                full_table_name = f"{schema}.{table_name}"
            else:
                full_table_name = table_name
            
            query = f"SELECT * FROM {full_table_name}"
            
            if where_clause:
                query += f" WHERE {where_clause}"
            
            if limit:
                query += f" LIMIT {limit}"
            
            if self.snowflake_conn:
                cursor = self.snowflake_conn.cursor()
                cursor.execute(query)
                df = cursor.fetch_pandas_all()
                cursor.close()
            else:
                df = pd.read_sql(query, self.engine)
            
            return df
        
        except Exception as e:
            raise RuntimeError(f"Failed to read table {table_name}: {str(e)}")
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute a custom SQL query and return results.
        
        Args:
            query: SQL query to execute
        
        Returns:
            Pandas DataFrame with query results
        """
        try:
            if self.snowflake_conn:
                cursor = self.snowflake_conn.cursor()
                cursor.execute(query)
                df = cursor.fetch_pandas_all()
                cursor.close()
            else:
                df = pd.read_sql(query, self.engine)
            
            return df
        
        except Exception as e:
            raise RuntimeError(f"Failed to execute query: {str(e)}")
    
    def get_table_info(self, table_name: str, schema: Optional[str] = None) -> Dict:
        """
        Get metadata about a table (row count, column names, data types).
        
        Args:
            table_name: Name of the table
            schema: Schema name (optional)
        
        Returns:
            Dictionary with table metadata
        """
        try:
            if schema:
                full_table_name = f"{schema}.{table_name}"
            else:
                full_table_name = table_name
            
            count_query = f"SELECT COUNT(*) as row_count FROM {full_table_name}"
            
            if self.snowflake_conn:
                cursor = self.snowflake_conn.cursor()
                cursor.execute(count_query)
                row_count = cursor.fetchone()[0]
                
                cursor.execute(f"SELECT * FROM {full_table_name} LIMIT 0")
                columns = [col[0] for col in cursor.description]
                cursor.close()
            else:
                with self.engine.connect() as conn:
                    result = conn.execute(text(count_query))
                    row_count = result.scalar()
                
                df_sample = pd.read_sql(f"SELECT * FROM {full_table_name} LIMIT 0", self.engine)
                columns = df_sample.columns.tolist()
            
            return {
                'table_name': table_name,
                'schema': schema,
                'row_count': int(row_count),
                'columns': columns,
                'column_count': len(columns)
            }
        
        except Exception as e:
            raise RuntimeError(f"Failed to get table info for {full_table_name}: {str(e)}")
    
    def test_connection(self) -> Tuple[bool, str]:
        """
        Test if the database connection is active.
        
        Returns:
            Tuple of (is_connected, message)
        """
        try:
            if self.snowflake_conn:
                cursor = self.snowflake_conn.cursor()
                cursor.execute("SELECT CURRENT_VERSION()")
                version = cursor.fetchone()[0]
                cursor.close()
                return True, f"Connected to Snowflake (version: {version})"
            
            elif self.engine:
                with self.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                    return True, "Database connection active"
            
            else:
                return False, "No active connection"
        
        except Exception as e:
            return False, f"Connection test failed: {str(e)}"
    
    def close(self):
        """Close database connection."""
        try:
            if self.snowflake_conn:
                self.snowflake_conn.close()
                self.snowflake_conn = None
            
            if self.engine:
                self.engine.dispose()
                self.engine = None
        
        except Exception as e:
            print(f"Error closing connection: {e}")
    
    def __del__(self):
        """Cleanup on object destruction."""
        self.close()
