"""
Tests for CLI functionality
"""

import pytest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

from cli import main, profile_file_command, profile_db_command, demo_command, formats_command


class TestCLI:
    """Test CLI functionality"""
    
    def setup_method(self):
        """Setup test data"""
        self.test_data = pd.DataFrame({
            'id': range(100),
            'name': [f'user_{i}' for i in range(100)],
            'age': np.random.randint(18, 80, 100),
            'score': np.random.normal(70, 15, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100)
        })
    
    def test_main_no_args(self):
        """Test main function with no arguments"""
        with patch('sys.argv', ['cli.py']):
            with patch('cli.argparse.ArgumentParser.print_help') as mock_help:
                main()
                mock_help.assert_called_once()
    
    def test_main_invalid_command(self):
        """Test main function with invalid command"""
        with patch('sys.argv', ['cli.py', 'invalid-command']):
            with patch('sys.exit') as mock_exit:
                main()
                mock_exit.assert_called_once_with(1)
    
    def test_formats_command(self):
        """Test formats command"""
        with patch('builtins.print') as mock_print:
            formats_command()
            mock_print.assert_called()
    
    @patch('cli.profile_dataframe')
    def test_profile_file_command_csv(self, mock_profile_dataframe):
        """Test profile-file command with CSV"""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.test_data.to_csv(f.name, index=False)
            csv_path = f.name
        
        try:
            # Mock the profile_dataframe function
            mock_profile_dataframe.return_value = {
                'column_profiles': [],
                'anomalies': [],
                'recommendations': []
            }
            
            # Create mock args
            args = Mock()
            args.file = csv_path
            args.output_dir = 'test_output'
            args.validation_rules = None
            args.table_name = None
            args.sample_size = None
            
            with patch('builtins.print') as mock_print:
                profile_file_command(args)
                
                # Verify profile_dataframe was called
                mock_profile_dataframe.assert_called_once()
                
                # Verify output messages
                mock_print.assert_called()
        
        finally:
            # Cleanup
            os.unlink(csv_path)
    
    @patch('cli.profile_dataframe')
    def test_profile_file_command_excel(self, mock_profile_dataframe):
        """Test profile-file command with Excel file"""
        # Create temporary Excel file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xlsx', delete=False) as f:
            # Note: This is a simplified test - in practice you'd need openpyxl
            excel_path = f.name
        
        try:
            # Mock the profile_dataframe function
            mock_profile_dataframe.return_value = {
                'column_profiles': [],
                'anomalies': [],
                'recommendations': []
            }
            
            # Create mock args
            args = Mock()
            args.file = excel_path
            args.output_dir = 'test_output'
            args.validation_rules = None
            args.table_name = None
            args.sample_size = None
            
            with patch('pandas.read_excel') as mock_read_excel:
                mock_read_excel.return_value = self.test_data
                
                with patch('builtins.print') as mock_print:
                    profile_file_command(args)
                    
                    # Verify profile_dataframe was called
                    mock_profile_dataframe.assert_called_once()
        
        finally:
            # Cleanup
            if os.path.exists(excel_path):
                os.unlink(excel_path)
    
    def test_profile_file_command_invalid_format(self):
        """Test profile-file command with invalid file format"""
        args = Mock()
        args.file = 'test.txt'
        args.output_dir = 'test_output'
        args.validation_rules = None
        args.table_name = None
        args.sample_size = None
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            profile_file_command(args)
    
    def test_profile_file_command_file_not_found(self):
        """Test profile-file command with non-existent file"""
        args = Mock()
        args.file = 'nonexistent_file.csv'
        args.output_dir = 'test_output'
        args.validation_rules = None
        args.table_name = None
        args.sample_size = None
        
        with pytest.raises(FileNotFoundError):
            profile_file_command(args)
    
    @patch('cli.profile_dataframe')
    def test_profile_file_command_with_validation_rules(self, mock_profile_dataframe):
        """Test profile-file command with validation rules"""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.test_data.to_csv(f.name, index=False)
            csv_path = f.name
        
        # Create temporary validation rules file
        validation_rules = {
            'age': {'min': 18, 'max': 80},
            'score': {'min': 0, 'max': 100}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(validation_rules, f)
            rules_path = f.name
        
        try:
            # Mock the profile_dataframe function
            mock_profile_dataframe.return_value = {
                'column_profiles': [],
                'anomalies': [],
                'recommendations': []
            }
            
            # Create mock args
            args = Mock()
            args.file = csv_path
            args.output_dir = 'test_output'
            args.validation_rules = rules_path
            args.table_name = None
            args.sample_size = None
            
            with patch('builtins.print') as mock_print:
                profile_file_command(args)
                
                # Verify profile_dataframe was called with validation rules
                call_args = mock_profile_dataframe.call_args
                assert call_args[1]['validation_rules'] == validation_rules
        
        finally:
            # Cleanup
            os.unlink(csv_path)
            os.unlink(rules_path)
    
    @patch('cli.profile_dataframe')
    def test_profile_file_command_with_sampling(self, mock_profile_dataframe):
        """Test profile-file command with sampling"""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.test_data.to_csv(f.name, index=False)
            csv_path = f.name
        
        try:
            # Mock the profile_dataframe function
            mock_profile_dataframe.return_value = {
                'column_profiles': [],
                'anomalies': [],
                'recommendations': []
            }
            
            # Create mock args
            args = Mock()
            args.file = csv_path
            args.output_dir = 'test_output'
            args.validation_rules = None
            args.table_name = None
            args.sample_size = 50
            
            with patch('builtins.print') as mock_print:
                profile_file_command(args)
                
                # Verify profile_dataframe was called
                mock_profile_dataframe.assert_called_once()
        
        finally:
            # Cleanup
            os.unlink(csv_path)
    
    @patch('cli.orchestrate')
    def test_profile_db_command(self, mock_orchestrate):
        """Test profile-db command"""
        # Create temporary config file
        config = {
            'cloud': 'aws',
            'connection': {
                'host': 'test-host',
                'database': 'test_db',
                'username': 'test_user',
                'password': 'test_password'
            },
            'tables': ['table1', 'table2'],
            'sample_size': 1000
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            config_path = f.name
        
        try:
            # Mock the orchestrate function
            mock_orchestrate.return_value = {
                'summary': {
                    'total_tables': 2,
                    'successful_tables': 2,
                    'failed_tables': 0
                }
            }
            
            # Create mock args
            args = Mock()
            args.config = config_path
            args.output_file = None
            
            with patch('builtins.print') as mock_print:
                profile_db_command(args)
                
                # Verify orchestrate was called with correct config
                mock_orchestrate.assert_called_once_with(config)
                
                # Verify output messages
                mock_print.assert_called()
        
        finally:
            # Cleanup
            os.unlink(config_path)
    
    def test_profile_db_command_config_not_found(self):
        """Test profile-db command with non-existent config file"""
        args = Mock()
        args.config = 'nonexistent_config.json'
        args.output_file = None
        
        with pytest.raises(FileNotFoundError):
            profile_db_command(args)
    
    @patch('cli.orchestrate')
    def test_profile_db_command_with_output_file(self, mock_orchestrate):
        """Test profile-db command with output file"""
        # Create temporary config file
        config = {
            'cloud': 'aws',
            'connection': {
                'host': 'test-host',
                'database': 'test_db',
                'username': 'test_user',
                'password': 'test_password'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            config_path = f.name
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name
        
        try:
            # Mock the orchestrate function
            mock_orchestrate.return_value = {
                'summary': {
                    'total_tables': 1,
                    'successful_tables': 1,
                    'failed_tables': 0
                }
            }
            
            # Create mock args
            args = Mock()
            args.config = config_path
            args.output_file = output_path
            
            with patch('builtins.print') as mock_print:
                profile_db_command(args)
                
                # Verify orchestrate was called
                mock_orchestrate.assert_called_once()
                
                # Verify output file was created
                assert os.path.exists(output_path)
        
        finally:
            # Cleanup
            os.unlink(config_path)
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    @patch('cli.orchestrate')
    def test_profile_db_command_with_save_results(self, mock_orchestrate):
        """Test profile-db command with save_results"""
        # Create temporary config file
        config = {
            'cloud': 'aws',
            'connection': {
                'host': 'test-host',
                'database': 'test_db',
                'username': 'test_user',
                'password': 'test_password'
            },
            'output_dir': 'test_output'
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            config_path = f.name
        
        try:
            # Mock the orchestrate function
            mock_orchestrate.return_value = {
                'summary': {
                    'total_tables': 1,
                    'successful_tables': 1,
                    'failed_tables': 0
                }
            }
            
            # Create mock args
            args = Mock()
            args.config = config_path
            args.output_file = None
            
            with patch('builtins.print') as mock_print:
                with patch('cli.save_results') as mock_save:
                    profile_db_command(args)
                    
                    # Verify save_results was called
                    mock_save.assert_called_once()
        
        finally:
            # Cleanup
            os.unlink(config_path)
    
    @patch('examples.demo.main')
    def test_demo_command(self, mock_demo_main):
        """Test demo command"""
        # Create mock args
        args = Mock()
        args.output_dir = 'demo_output'
        
        with patch('builtins.print') as mock_print:
            with patch('os.environ') as mock_env:
                demo_command(args)
                
                # Verify demo main was called
                mock_demo_main.assert_called_once()
                
                # Verify environment variable was set
                mock_env.__setitem__.assert_called_with('DEMO_OUTPUT_DIR', 'demo_output')
    
    def test_cli_error_handling(self):
        """Test CLI error handling"""
        # Test with invalid file format
        args = Mock()
        args.file = 'test.invalid'
        args.output_dir = 'test_output'
        args.validation_rules = None
        args.table_name = None
        args.sample_size = None
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            profile_file_command(args)
    
    @patch('cli.profile_dataframe')
    def test_cli_file_loading_error(self, mock_profile_dataframe):
        """Test CLI file loading error handling"""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("invalid,csv,content\nwith,wrong,format")
            csv_path = f.name
        
        try:
            # Create mock args
            args = Mock()
            args.file = csv_path
            args.output_dir = 'test_output'
            args.validation_rules = None
            args.table_name = None
            args.sample_size = None
            
            with patch('pandas.read_csv') as mock_read_csv:
                mock_read_csv.side_effect = Exception("CSV parsing error")
                
                with pytest.raises(ValueError, match="Error loading file"):
                    profile_file_command(args)
        
        finally:
            # Cleanup
            os.unlink(csv_path)


class TestCLIIntegration:
    """Integration tests for CLI"""
    
    def test_cli_with_real_data(self):
        """Test CLI with real data processing"""
        # Create test data
        test_data = pd.DataFrame({
            'id': range(50),
            'value': np.random.normal(0, 1, 50),
            'category': ['A', 'B'] * 25
        })
        
        # Create temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create CSV file
            csv_path = os.path.join(temp_dir, 'test_data.csv')
            test_data.to_csv(csv_path, index=False)
            
            # Create validation rules
            rules = {
                'id': {'unique': True},
                'value': {'min': -3, 'max': 3}
            }
            rules_path = os.path.join(temp_dir, 'rules.json')
            with open(rules_path, 'w') as f:
                json.dump(rules, f)
            
            # Create mock args
            args = Mock()
            args.file = csv_path
            args.output_dir = temp_dir
            args.validation_rules = rules_path
            args.table_name = 'test_table'
            args.sample_size = None
            
            with patch('cli.profile_dataframe') as mock_profile:
                mock_profile.return_value = {
                    'column_profiles': [],
                    'anomalies': [],
                    'recommendations': []
                }
                
                with patch('builtins.print'):
                    profile_file_command(args)
                    
                    # Verify profile_dataframe was called with correct parameters
                    call_args = mock_profile.call_args
                    assert call_args[1]['table_name'] == 'test_table'
                    assert call_args[1]['validation_rules'] == rules
                    assert call_args[1]['output_dir'] == temp_dir


if __name__ == "__main__":
    pytest.main([__file__]) 