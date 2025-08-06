"""
Base cloud connector class
"""

from abc import ABC, abstractmethod
from typing import Dict
from sqlalchemy import Engine


class CloudConnector(ABC):
    """Base class for cloud database connectors"""
    
    def __init__(self, config: Dict):
        """
        Initialize the connector with configuration
        
        Args:
            config: Dictionary containing connection parameters
        """
        self.config = config
    
    @abstractmethod
    def get_engine(self) -> Engine:
        """
        Get SQLAlchemy engine for the database connection
        
        Returns:
            SQLAlchemy Engine instance
        """
        raise NotImplementedError
    
    def test_connection(self) -> bool:
        """
        Test the database connection
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            engine = self.get_engine()
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False 