"""
Tests for cloud connectors
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from data_profiler.connectors import (
    CloudConnector,
    AWSConnector,
    GCPConnector,
    AzureConnector
)


class TestCloudConnector:
    """Test base CloudConnector class"""
    
    def test_abstract_methods(self):
        """Test that CloudConnector is abstract"""
        with pytest.raises(TypeError):
            CloudConnector({})


class TestAWSConnector:
    """Test AWS connector"""
    
    def setup_method(self):
        """Setup test configuration"""
        self.config = {
            'host': 'test-host.amazonaws.com',
            'port': 5439,
            'database': 'test_db',
            'username': 'test_user',
            'password': 'test_password',
            'db_type': 'redshift'
        }
        self.connector = AWSConnector(self.config)
    
    @patch('data_profiler.connectors.aws.create_engine')
    def test_get_engine_redshift(self, mock_create_engine):
        """Test Redshift engine creation"""
        mock_engine = Mock(spec=Engine)
        mock_create_engine.return_value = mock_engine
        
        engine = self.connector.get_engine()
        
        mock_create_engine.assert_called_once()
        call_args = mock_create_engine.call_args[0][0]
        assert 'postgresql://' in call_args
        assert 'test-host.amazonaws.com' in call_args
        assert 'test_db' in call_args
    
    @patch('data_profiler.connectors.aws.create_engine')
    def test_get_engine_rds_postgresql(self, mock_create_engine):
        """Test RDS PostgreSQL engine creation"""
        config = self.config.copy()
        config['db_type'] = 'rds'
        config['rds_engine'] = 'postgresql'
        
        connector = AWSConnector(config)
        mock_engine = Mock(spec=Engine)
        mock_create_engine.return_value = mock_engine
        
        engine = connector.get_engine()
        
        mock_create_engine.assert_called_once()
        call_args = mock_create_engine.call_args[0][0]
        assert 'postgresql://' in call_args
    
    @patch('data_profiler.connectors.aws.create_engine')
    def test_get_engine_rds_mysql(self, mock_create_engine):
        """Test RDS MySQL engine creation"""
        config = self.config.copy()
        config['db_type'] = 'rds'
        config['rds_engine'] = 'mysql'
        
        connector = AWSConnector(config)
        mock_engine = Mock(spec=Engine)
        mock_create_engine.return_value = mock_engine
        
        engine = connector.get_engine()
        
        mock_create_engine.assert_called_once()
        call_args = mock_create_engine.call_args[0][0]
        assert 'mysql+pymysql://' in call_args
    
    def test_get_engine_invalid_db_type(self):
        """Test invalid database type"""
        config = self.config.copy()
        config['db_type'] = 'invalid'
        
        connector = AWSConnector(config)
        
        with pytest.raises(ValueError, match="Unsupported AWS database type"):
            connector.get_engine()
    
    def test_missing_required_parameters(self):
        """Test missing required parameters"""
        config = {
            'host': 'test-host',
            # Missing database, username, password
        }
        
        connector = AWSConnector(config)
        
        with pytest.raises(ValueError, match="Missing required connection parameters"):
            connector.get_engine()
    
    @patch('data_profiler.connectors.aws.create_engine')
    def test_get_tables_redshift(self, mock_create_engine):
        """Test getting tables from Redshift"""
        mock_engine = Mock(spec=Engine)
        mock_conn = Mock()
        mock_result = Mock()
        mock_result.fetchall.return_value = [('table1',), ('table2',)]
        
        mock_conn.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_create_engine.return_value = mock_engine
        
        tables = self.connector.get_tables()
        
        assert tables == ['table1', 'table2']
        mock_conn.execute.assert_called_once()
    
    @patch('data_profiler.connectors.aws.create_engine')
    def test_test_connection_success(self, mock_create_engine):
        """Test successful connection test"""
        mock_engine = Mock(spec=Engine)
        mock_conn = Mock()
        mock_result = Mock()
        
        mock_conn.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_create_engine.return_value = mock_engine
        
        result = self.connector.test_connection()
        
        assert result is True
        mock_conn.execute.assert_called_once_with("SELECT 1")
    
    @patch('data_profiler.connectors.aws.create_engine')
    def test_test_connection_failure(self, mock_create_engine):
        """Test failed connection test"""
        mock_engine = Mock(spec=Engine)
        mock_engine.connect.side_effect = Exception("Connection failed")
        mock_create_engine.return_value = mock_engine
        
        result = self.connector.test_connection()
        
        assert result is False


class TestGCPConnector:
    """Test GCP connector"""
    
    def setup_method(self):
        """Setup test configuration"""
        self.config = {
            'project_id': 'test-project',
            'dataset_id': 'test_dataset',
            'credentials_path': '/path/to/credentials.json'
        }
        self.connector = GCPConnector(self.config)
    
    @patch('data_profiler.connectors.gcp.create_engine')
    def test_get_engine_with_credentials(self, mock_create_engine):
        """Test BigQuery engine creation with credentials"""
        mock_engine = Mock(spec=Engine)
        mock_create_engine.return_value = mock_engine
        
        engine = self.connector.get_engine()
        
        mock_create_engine.assert_called_once()
        call_args = mock_create_engine.call_args[0][0]
        assert 'bigquery://' in call_args
        assert 'test-project' in call_args
        assert 'test_dataset' in call_args
        assert 'credentials_path' in call_args
    
    @patch('data_profiler.connectors.gcp.create_engine')
    def test_get_engine_without_credentials(self, mock_create_engine):
        """Test BigQuery engine creation without credentials"""
        config = {
            'project_id': 'test-project',
            'dataset_id': 'test_dataset'
        }
        connector = GCPConnector(config)
        
        mock_engine = Mock(spec=Engine)
        mock_create_engine.return_value = mock_engine
        
        engine = connector.get_engine()
        
        mock_create_engine.assert_called_once()
        call_args = mock_create_engine.call_args[0][0]
        assert 'bigquery://' in call_args
        assert 'credentials_path' not in call_args
    
    def test_missing_required_parameters(self):
        """Test missing required parameters"""
        config = {
            'project_id': 'test-project'
            # Missing dataset_id
        }
        
        connector = GCPConnector(config)
        
        with pytest.raises(ValueError, match="Missing required connection parameters"):
            connector.get_engine()
    
    @patch('data_profiler.connectors.gcp.create_engine')
    def test_get_tables(self, mock_create_engine):
        """Test getting tables from BigQuery"""
        mock_engine = Mock(spec=Engine)
        mock_conn = Mock()
        mock_result = Mock()
        mock_result.fetchall.return_value = [('table1',), ('table2',)]
        
        mock_conn.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_create_engine.return_value = mock_engine
        
        tables = self.connector.get_tables()
        
        assert tables == ['table1', 'table2']
        mock_conn.execute.assert_called_once()
    
    @patch('data_profiler.connectors.gcp.create_engine')
    def test_get_table_info(self, mock_create_engine):
        """Test getting table information"""
        mock_engine = Mock(spec=Engine)
        mock_conn = Mock()
        mock_result = Mock()
        mock_result.fetchone.return_value = ('test_table', 1000, 1024, '2023-01-01', '2023-01-02')
        
        mock_conn.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_create_engine.return_value = mock_engine
        
        info = self.connector.get_table_info('test_table')
        
        assert info['table_id'] == 'test_table'
        assert info['row_count'] == 1000
        assert info['size_bytes'] == 1024


class TestAzureConnector:
    """Test Azure connector"""
    
    def setup_method(self):
        """Setup test configuration"""
        self.config = {
            'server': 'test-server.database.windows.net',
            'database': 'test_db',
            'username': 'test_user',
            'password': 'test_password',
            'port': 1433,
            'db_type': 'sql_database'
        }
        self.connector = AzureConnector(self.config)
    
    @patch('data_profiler.connectors.azure.pyodbc')
    @patch('data_profiler.connectors.azure.create_engine')
    def test_get_engine_sql_database(self, mock_create_engine, mock_pyodbc):
        """Test SQL Database engine creation"""
        mock_engine = Mock(spec=Engine)
        mock_create_engine.return_value = mock_engine
        
        engine = self.connector.get_engine()
        
        mock_create_engine.assert_called_once()
        call_args = mock_create_engine.call_args[0][0]
        assert 'mssql+pyodbc://' in call_args
        assert 'test-server.database.windows.net' in call_args
        assert 'test_db' in call_args
    
    @patch('data_profiler.connectors.azure.pyodbc', None)
    def test_get_engine_missing_pyodbc(self):
        """Test engine creation without pyodbc"""
        with pytest.raises(ImportError, match="pyodbc is required"):
            self.connector.get_engine()
    
    def test_get_engine_invalid_db_type(self):
        """Test invalid database type"""
        config = self.config.copy()
        config['db_type'] = 'invalid'
        
        connector = AzureConnector(config)
        
        with pytest.raises(ValueError, match="Unsupported Azure database type"):
            connector.get_engine()
    
    @patch('data_profiler.connectors.azure.pyodbc')
    @patch('data_profiler.connectors.azure.create_engine')
    def test_get_tables(self, mock_create_engine, mock_pyodbc):
        """Test getting tables from Azure SQL"""
        mock_engine = Mock(spec=Engine)
        mock_conn = Mock()
        mock_result = Mock()
        mock_result.fetchall.return_value = [('table1',), ('table2',)]
        
        mock_conn.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_create_engine.return_value = mock_engine
        
        tables = self.connector.get_tables()
        
        assert tables == ['table1', 'table2']
        mock_conn.execute.assert_called_once()
    
    @patch('data_profiler.connectors.azure.pyodbc')
    @patch('data_profiler.connectors.azure.create_engine')
    def test_get_table_info(self, mock_create_engine, mock_pyodbc):
        """Test getting table information"""
        mock_engine = Mock(spec=Engine)
        mock_conn = Mock()
        mock_result = Mock()
        mock_result.fetchone.return_value = ('test_table', 1000, 1024.5)
        
        mock_conn.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_create_engine.return_value = mock_engine
        
        info = self.connector.get_table_info('test_table')
        
        assert info['table_name'] == 'test_table'
        assert info['row_count'] == 1000
        assert info['size_mb'] == 1024.5


class TestConnectorIntegration:
    """Integration tests for connectors"""
    
    def test_connector_factory_pattern(self):
        """Test that connectors can be created from config"""
        from data_profiler.orchestrator import get_connector
        
        # Test AWS connector
        aws_config = {
            'host': 'test-host',
            'database': 'test_db',
            'username': 'test_user',
            'password': 'test_password'
        }
        
        with patch('data_profiler.connectors.aws.create_engine'):
            aws_connector = get_connector('aws', aws_config)
            assert isinstance(aws_connector, AWSConnector)
        
        # Test GCP connector
        gcp_config = {
            'project_id': 'test-project',
            'dataset_id': 'test_dataset'
        }
        
        with patch('data_profiler.connectors.gcp.create_engine'):
            gcp_connector = get_connector('gcp', gcp_config)
            assert isinstance(gcp_connector, GCPConnector)
        
        # Test Azure connector
        azure_config = {
            'server': 'test-server',
            'database': 'test_db',
            'username': 'test_user',
            'password': 'test_password'
        }
        
        with patch('data_profiler.connectors.azure.pyodbc'), \
             patch('data_profiler.connectors.azure.create_engine'):
            azure_connector = get_connector('azure', azure_config)
            assert isinstance(azure_connector, AzureConnector)
        
        # Test invalid provider
        with pytest.raises(ValueError, match="Unsupported cloud provider"):
            get_connector('invalid', {})
    
    def test_connector_error_handling(self):
        """Test error handling in connectors"""
        # Test with invalid configuration
        invalid_config = {}
        
        with pytest.raises(ValueError):
            AWSConnector(invalid_config).get_engine()
        
        with pytest.raises(ValueError):
            GCPConnector(invalid_config).get_engine()
        
        with patch('data_profiler.connectors.azure.pyodbc', None):
            with pytest.raises(ImportError):
                AzureConnector(invalid_config).get_engine()


if __name__ == "__main__":
    pytest.main([__file__]) 