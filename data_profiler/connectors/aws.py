"""
AWS cloud connector for Redshift and RDS
"""

from typing import Dict
from sqlalchemy import create_engine, Engine
from sqlalchemy.engine import URL
from .base import CloudConnector


class AWSConnector(CloudConnector):
    """AWS connector for Redshift and RDS databases"""
    
    def get_engine(self) -> Engine:
        """
        Create SQLAlchemy engine for AWS database
        
        Returns:
            SQLAlchemy Engine instance
        """
        # Extract connection parameters
        host = self.config.get('host')
        port = self.config.get('port', 5439)
        database = self.config.get('database')
        username = self.config.get('username')
        password = self.config.get('password')
        
        # Validate required parameters
        if not all([host, database, username, password]):
            raise ValueError("Missing required connection parameters: host, database, username, and password are required")
        
        # Determine database type and driver
        db_type = self.config.get('db_type', 'redshift')
        
        if db_type.lower() == 'redshift':
            # Redshift connection
            connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"
            return create_engine(
                connection_string,
                echo=self.config.get('echo', False),
                pool_size=self.config.get('pool_size', 5),
                max_overflow=self.config.get('max_overflow', 10)
            )
        elif db_type.lower() == 'rds':
            # RDS PostgreSQL/MySQL connection
            db_engine = self.config.get('rds_engine', 'postgresql')
            if db_engine.lower() == 'mysql':
                connection_string = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
            else:
                connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"
            
            return create_engine(
                connection_string,
                echo=self.config.get('echo', False),
                pool_size=self.config.get('pool_size', 5),
                max_overflow=self.config.get('max_overflow', 10)
            )
        else:
            raise ValueError(f"Unsupported AWS database type: {db_type}")
    
    def get_tables(self) -> list:
        """
        Get list of tables in the database
        
        Returns:
            List of table names
        """
        engine = self.get_engine()
        db_type = self.config.get('db_type', 'redshift')
        
        if db_type.lower() == 'redshift':
            query = """
                SELECT tablename 
                FROM pg_tables 
                WHERE schemaname = 'public'
                ORDER BY tablename
            """
        else:
            query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """
        
        with engine.connect() as conn:
            result = conn.execute(query)
            return [row[0] for row in result.fetchall()] 