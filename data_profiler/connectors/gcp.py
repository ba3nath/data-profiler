"""
GCP cloud connector for BigQuery
"""

from typing import Dict
from sqlalchemy import create_engine, Engine
from .base import CloudConnector


class GCPConnector(CloudConnector):
    """GCP connector for BigQuery"""
    
    def get_engine(self) -> Engine:
        """
        Create SQLAlchemy engine for BigQuery
        
        Returns:
            SQLAlchemy Engine instance
        """
        # Extract connection parameters
        project_id = self.config.get('project_id')
        dataset_id = self.config.get('dataset_id')
        credentials_path = self.config.get('credentials_path')
        
        if credentials_path:
            # Use service account credentials file
            connection_string = f"bigquery://{project_id}/{dataset_id}?credentials_path={credentials_path}"
        else:
            # Use default credentials (Application Default Credentials)
            connection_string = f"bigquery://{project_id}/{dataset_id}"
        
        return create_engine(
            connection_string,
            echo=self.config.get('echo', False),
            pool_size=self.config.get('pool_size', 5),
            max_overflow=self.config.get('max_overflow', 10)
        )
    
    def get_tables(self) -> list:
        """
        Get list of tables in the BigQuery dataset
        
        Returns:
            List of table names
        """
        engine = self.get_engine()
        project_id = self.config.get('project_id')
        dataset_id = self.config.get('dataset_id')
        
        query = f"""
            SELECT table_id 
            FROM `{project_id}.{dataset_id}.__TABLES__`
            ORDER BY table_id
        """
        
        with engine.connect() as conn:
            result = conn.execute(query)
            return [row[0] for row in result.fetchall()]
    
    def get_table_info(self, table_name: str) -> Dict:
        """
        Get detailed information about a BigQuery table
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary containing table metadata
        """
        engine = self.get_engine()
        project_id = self.config.get('project_id')
        dataset_id = self.config.get('dataset_id')
        
        query = f"""
            SELECT 
                table_id,
                row_count,
                size_bytes,
                created,
                modified
            FROM `{project_id}.{dataset_id}.__TABLES__`
            WHERE table_id = '{table_name}'
        """
        
        with engine.connect() as conn:
            result = conn.execute(query)
            row = result.fetchone()
            if row:
                return {
                    'table_id': row[0],
                    'row_count': row[1],
                    'size_bytes': row[2],
                    'created': row[3],
                    'modified': row[4]
                }
            return {} 