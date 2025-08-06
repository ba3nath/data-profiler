"""
Azure cloud connector for SQL Database and Synapse
"""

from typing import Dict
from sqlalchemy import create_engine, Engine
from .base import CloudConnector


class AzureConnector(CloudConnector):
    """Azure connector for SQL Database and Synapse"""
    
    def get_engine(self) -> Engine:
        """
        Create SQLAlchemy engine for Azure database
        
        Returns:
            SQLAlchemy Engine instance
        """
        # Extract connection parameters
        server = self.config.get('server')
        database = self.config.get('database')
        username = self.config.get('username')
        password = self.config.get('password')
        port = self.config.get('port', 1433)
        
        # Determine database type
        db_type = self.config.get('db_type', 'sql_database')
        
        if db_type.lower() in ['sql_database', 'synapse']:
            # Azure SQL Database or Synapse connection
            connection_string = f"mssql+pyodbc://{username}:{password}@{server}:{port}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
            
            return create_engine(
                connection_string,
                echo=self.config.get('echo', False),
                pool_size=self.config.get('pool_size', 5),
                max_overflow=self.config.get('max_overflow', 10)
            )
        else:
            raise ValueError(f"Unsupported Azure database type: {db_type}")
    
    def get_tables(self) -> list:
        """
        Get list of tables in the Azure database
        
        Returns:
            List of table names
        """
        engine = self.get_engine()
        
        query = """
            SELECT TABLE_NAME 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_TYPE = 'BASE TABLE'
            ORDER BY TABLE_NAME
        """
        
        with engine.connect() as conn:
            result = conn.execute(query)
            return [row[0] for row in result.fetchall()]
    
    def get_table_info(self, table_name: str) -> Dict:
        """
        Get detailed information about an Azure table
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary containing table metadata
        """
        engine = self.get_engine()
        
        query = f"""
            SELECT 
                t.name as table_name,
                p.rows as row_count,
                CAST(ROUND((SUM(a.total_pages) * 8) / 1024.00, 2) AS NUMERIC(36, 2)) AS size_mb
            FROM sys.tables t
            INNER JOIN sys.indexes i ON t.OBJECT_ID = i.object_id
            INNER JOIN sys.partitions p ON i.object_id = p.OBJECT_ID AND i.index_id = p.index_id
            INNER JOIN sys.allocation_units a ON p.partition_id = a.container_id
            WHERE t.name = '{table_name}'
            GROUP BY t.name, p.rows
        """
        
        with engine.connect() as conn:
            result = conn.execute(query)
            row = result.fetchone()
            if row:
                return {
                    'table_name': row[0],
                    'row_count': row[1],
                    'size_mb': row[2]
                }
            return {} 