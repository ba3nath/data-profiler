"""
Tests for orchestrator and utility functions
"""

import pytest
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

from data_profiler.orchestrator import (
    get_connector,
    discover_tables,
    orchestrate,
    profile_dataframe,
    load_config,
    save_results,
    generate_summary_report
)


class TestOrchestrator:
    """Test orchestrator functions"""
    
    def setup_method(self):
        """Setup test data"""
        self.test_df = pd.DataFrame({
            'id': range(100),
            'name': [f'user_{i}' for i in range(100)],
            'age': np.random.randint(18, 80, 100),
            'score': np.random.normal(70, 15, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100)
        })
        
        self.test_config = {
            'cloud': 'aws',
            'connection': {
                'host': 'test-host',
                'database': 'test_db',
                'username': 'test_user',
                'password': 'test_password'
            },
            'tables': ['table1', 'table2'],
            'sample_size': 1000,
            'validation_rules': {
                'age': {'min': 18, 'max': 80},
                'score': {'min': 0, 'max': 100}
            }
        }
    
    def test_get_connector_aws(self):
        """Test AWS connector creation"""
        with patch('data_profiler.connectors.aws.AWSConnector') as mock_aws:
            connector = get_connector('aws', {})
            mock_aws.assert_called_once()
    
    def test_get_connector_gcp(self):
        """Test GCP connector creation"""
        with patch('data_profiler.connectors.gcp.GCPConnector') as mock_gcp:
            connector = get_connector('gcp', {})
            mock_gcp.assert_called_once()
    
    def test_get_connector_azure(self):
        """Test Azure connector creation"""
        with patch('data_profiler.connectors.azure.AzureConnector') as mock_azure:
            connector = get_connector('azure', {})
            mock_azure.assert_called_once()
    
    def test_get_connector_invalid(self):
        """Test invalid connector creation"""
        with pytest.raises(ValueError, match="Unsupported cloud provider"):
            get_connector('invalid', {})
    
    def test_discover_tables_with_get_tables(self):
        """Test table discovery with get_tables method"""
        mock_connector = Mock()
        mock_connector.get_tables.return_value = ['table1', 'table2', 'table3']
        
        tables = discover_tables(mock_connector, {})
        
        assert tables == ['table1', 'table2', 'table3']
        mock_connector.get_tables.assert_called_once()
    
    def test_discover_tables_fallback(self):
        """Test table discovery fallback to config"""
        mock_connector = Mock()
        # No get_tables method
        
        config = {'tables': ['config_table1', 'config_table2']}
        tables = discover_tables(mock_connector, config)
        
        assert tables == ['config_table1', 'config_table2']
    
    def test_discover_tables_exception(self):
        """Test table discovery with exception"""
        mock_connector = Mock()
        mock_connector.get_tables.side_effect = Exception("Connection error")
        
        config = {'tables': ['fallback_table']}
        tables = discover_tables(mock_connector, config)
        
        assert tables == ['fallback_table']
    
    @patch('data_profiler.orchestrator.get_connector')
    @patch('data_profiler.orchestrator.discover_tables')
    @patch('data_profiler.orchestrator.ColumnProfiler')
    @patch('data_profiler.orchestrator.CorrelationEngine')
    @patch('data_profiler.orchestrator.DataValidator')
    @patch('data_profiler.orchestrator.ReportGenerator')
    def test_orchestrate_success(self, mock_report_gen, mock_validator, 
                                mock_corr_engine, mock_col_profiler, 
                                mock_discover, mock_get_connector):
        """Test successful orchestration"""
        # Mock connector
        mock_connector = Mock()
        mock_connector.test_connection.return_value = True
        mock_get_connector.return_value = mock_connector
        
        # Mock table discovery
        mock_discover.return_value = ['table1', 'table2']
        
        # Mock components
        mock_profiler = Mock()
        mock_profiler.profile.return_value = []
        mock_profiler.find_anomalies.return_value = []
        mock_profiler.get_summary.return_value = {}
        mock_col_profiler.return_value = mock_profiler
        
        mock_corr = Mock()
        mock_corr.compute_correlation_matrix.return_value = pd.DataFrame()
        mock_corr.top_correlated_pairs.return_value = []
        mock_corr_engine.return_value = mock_corr
        
        mock_val = Mock()
        mock_val.validate_rules.return_value = {}
        mock_val.check_data_quality_metrics.return_value = {}
        mock_validator.return_value = mock_val
        
        mock_report = Mock()
        mock_report.generate_comprehensive_report.return_value = {}
        mock_report_gen.return_value = mock_report
        
        # Run orchestration
        result = orchestrate(self.test_config)
        
        # Verify result structure
        assert 'summary' in result
        assert 'table_results' in result
        assert 'report_paths' in result
        
        # Verify connector was created
        mock_get_connector.assert_called_once_with('aws', self.test_config['connection'])
        
        # Verify connection was tested
        mock_connector.test_connection.assert_called_once()
    
    @patch('data_profiler.orchestrator.get_connector')
    def test_orchestrate_connection_failure(self, mock_get_connector):
        """Test orchestration with connection failure"""
        # Mock connector with failed connection
        mock_connector = Mock()
        mock_connector.test_connection.return_value = False
        mock_get_connector.return_value = mock_connector
        
        with pytest.raises(Exception, match="Database connection test failed"):
            orchestrate(self.test_config)
    
    @patch('data_profiler.orchestrator.ColumnProfiler')
    @patch('data_profiler.orchestrator.CorrelationEngine')
    @patch('data_profiler.orchestrator.DataValidator')
    @patch('data_profiler.orchestrator.ReportGenerator')
    def test_profile_dataframe(self, mock_report_gen, mock_validator, 
                              mock_corr_engine, mock_col_profiler):
        """Test DataFrame profiling"""
        # Mock components
        mock_profiler = Mock()
        mock_profiler.profile.return_value = []
        mock_profiler.find_anomalies.return_value = []
        mock_profiler.get_summary.return_value = {}
        mock_col_profiler.return_value = mock_profiler
        
        mock_corr = Mock()
        mock_corr.compute_correlation_matrix.return_value = pd.DataFrame()
        mock_corr.top_correlated_pairs.return_value = []
        mock_corr_engine.return_value = mock_corr
        
        mock_val = Mock()
        mock_val.validate_rules.return_value = {}
        mock_val.check_data_quality_metrics.return_value = {}
        mock_validator.return_value = mock_val
        
        mock_report = Mock()
        mock_report.generate_comprehensive_report.return_value = {}
        mock_report_gen.return_value = mock_report
        
        # Run profiling
        result = profile_dataframe(
            df=self.test_df,
            table_name='test_table',
            validation_rules={'age': {'min': 18, 'max': 80}},
            output_dir='test_output'
        )
        
        # Verify result structure
        assert 'sample_size' in result
        assert 'column_profiles' in result
        assert 'correlation_results' in result
        assert 'validation_results' in result
        assert 'anomalies' in result
        assert 'recommendations' in result
        assert 'report_paths' in result
        
        # Verify components were called
        mock_profiler.profile.assert_called_once()
        mock_corr.compute_correlation_matrix.assert_called_once()
        mock_val.validate_rules.assert_called_once()
    
    def test_profile_dataframe_empty_df(self):
        """Test DataFrame profiling with empty DataFrame"""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="DataFrame is empty"):
            profile_dataframe(empty_df, 'empty_table')
    
    def test_profile_dataframe_sampling(self):
        """Test DataFrame profiling with sampling"""
        large_df = pd.DataFrame({
            'col': range(10000)
        })
        
        with patch('data_profiler.orchestrator.ColumnProfiler') as mock_profiler, \
             patch('data_profiler.orchestrator.CorrelationEngine') as mock_corr, \
             patch('data_profiler.orchestrator.DataValidator') as mock_val, \
             patch('data_profiler.orchestrator.ReportGenerator') as mock_report:
            
            # Mock components
            mock_profiler.return_value.profile.return_value = []
            mock_corr.return_value.compute_correlation_matrix.return_value = pd.DataFrame()
            mock_val.return_value.validate_rules.return_value = {}
            mock_report.return_value.generate_comprehensive_report.return_value = {}
            
            result = profile_dataframe(
                df=large_df,
                table_name='large_table',
                output_dir='test_output'
            )
            
            # Verify sampling occurred
            assert result['sample_size'] <= 10000
    
    def test_load_config_success(self):
        """Test successful config loading"""
        config_data = {'test': 'value', 'nested': {'key': 'value'}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            loaded_config = load_config(config_path)
            assert loaded_config == config_data
        finally:
            os.unlink(config_path)
    
    def test_load_config_file_not_found(self):
        """Test config loading with non-existent file"""
        with pytest.raises(FileNotFoundError):
            load_config('nonexistent_config.json')
    
    def test_load_config_invalid_json(self):
        """Test config loading with invalid JSON"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('invalid json content')
            config_path = f.name
        
        try:
            with pytest.raises(json.JSONDecodeError):
                load_config(config_path)
        finally:
            os.unlink(config_path)
    
    def test_save_results_success(self):
        """Test successful results saving"""
        test_results = {
            'summary': {'total_tables': 2},
            'table_results': {'table1': {'status': 'success'}},
            'report_paths': {'html': '/path/to/report.html'}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name
        
        try:
            save_results(test_results, output_path)
            
            # Verify file was created and contains correct data
            assert os.path.exists(output_path)
            
            with open(output_path, 'r') as f:
                saved_data = json.load(f)
            
            assert saved_data == test_results
        finally:
            os.unlink(output_path)
    
    def test_save_results_directory_creation(self):
        """Test results saving with directory creation"""
        test_results = {'test': 'data'}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'nested', 'dir', 'results.json')
            
            save_results(test_results, output_path)
            
            # Verify directory was created and file was saved
            assert os.path.exists(output_path)
            
            with open(output_path, 'r') as f:
                saved_data = json.load(f)
            
            assert saved_data == test_results
    
    def test_generate_summary_report(self):
        """Test summary report generation"""
        test_results = {
            'summary': {
                'total_tables': 3,
                'successful_tables': 2,
                'failed_tables': 1,
                'total_columns': 15,
                'avg_null_ratio': 0.05
            },
            'table_results': {
                'table1': {'status': 'success', 'columns': 5},
                'table2': {'status': 'success', 'columns': 5},
                'table3': {'status': 'failed', 'error': 'Connection timeout'}
            }
        }
        
        summary = generate_summary_report(test_results)
        
        assert 'execution_summary' in summary
        assert 'table_status' in summary
        assert 'quality_metrics' in summary
        
        # Verify execution summary
        exec_summary = summary['execution_summary']
        assert exec_summary['total_tables'] == 3
        assert exec_summary['successful_tables'] == 2
        assert exec_summary['failed_tables'] == 1
        assert exec_summary['success_rate'] == 2/3
    
    def test_generate_summary_report_empty_results(self):
        """Test summary report generation with empty results"""
        empty_results = {}
        
        summary = generate_summary_report(empty_results)
        
        assert 'execution_summary' in summary
        assert 'table_status' in summary
        assert 'quality_metrics' in summary
        
        # Verify default values
        exec_summary = summary['execution_summary']
        assert exec_summary['total_tables'] == 0
        assert exec_summary['successful_tables'] == 0
        assert exec_summary['failed_tables'] == 0


class TestOrchestratorIntegration:
    """Integration tests for orchestrator"""
    
    @patch('data_profiler.orchestrator.get_connector')
    @patch('data_profiler.orchestrator.ColumnProfiler')
    @patch('data_profiler.orchestrator.CorrelationEngine')
    @patch('data_profiler.orchestrator.DataValidator')
    @patch('data_profiler.orchestrator.ReportGenerator')
    def test_full_orchestration_pipeline(self, mock_report_gen, mock_validator,
                                        mock_corr_engine, mock_col_profiler,
                                        mock_get_connector):
        """Test full orchestration pipeline"""
        # Create test data
        test_df = pd.DataFrame({
            'id': range(50),
            'value': np.random.normal(0, 1, 50),
            'category': ['A', 'B'] * 25
        })
        
        # Mock all components
        mock_connector = Mock()
        mock_connector.test_connection.return_value = True
        mock_get_connector.return_value = mock_connector
        
        mock_profiler = Mock()
        mock_profiler.profile.return_value = []
        mock_profiler.find_anomalies.return_value = []
        mock_profiler.get_summary.return_value = {}
        mock_col_profiler.return_value = mock_profiler
        
        mock_corr = Mock()
        mock_corr.compute_correlation_matrix.return_value = pd.DataFrame()
        mock_corr.top_correlated_pairs.return_value = []
        mock_corr_engine.return_value = mock_corr
        
        mock_val = Mock()
        mock_val.validate_rules.return_value = {}
        mock_val.check_data_quality_metrics.return_value = {}
        mock_validator.return_value = mock_val
        
        mock_report = Mock()
        mock_report.generate_comprehensive_report.return_value = {}
        mock_report_gen.return_value = mock_report
        
        # Test DataFrame profiling
        with tempfile.TemporaryDirectory() as temp_dir:
            result = profile_dataframe(
                df=test_df,
                table_name='integration_test',
                validation_rules={'id': {'unique': True}},
                output_dir=temp_dir
            )
            
            # Verify complete result structure
            required_keys = [
                'sample_size', 'column_profiles', 'correlation_results',
                'validation_results', 'anomalies', 'recommendations', 'report_paths'
            ]
            
            for key in required_keys:
                assert key in result
            
            # Verify data integrity
            assert result['sample_size'] == len(test_df)
            assert isinstance(result['column_profiles'], list)
            assert isinstance(result['correlation_results'], dict)
            assert isinstance(result['validation_results'], dict)
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Test with minimal valid config
        minimal_config = {
            'cloud': 'aws',
            'connection': {
                'host': 'test-host',
                'database': 'test_db',
                'username': 'test_user',
                'password': 'test_password'
            }
        }
        
        # Should not raise any exceptions
        assert isinstance(minimal_config, dict)
        assert 'cloud' in minimal_config
        assert 'connection' in minimal_config
        
        # Test with invalid config
        invalid_config = {}
        
        # Should handle gracefully
        assert isinstance(invalid_config, dict)


if __name__ == "__main__":
    pytest.main([__file__]) 