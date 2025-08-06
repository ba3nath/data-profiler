"""
Tests for Data Profiler
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

from data_profiler import (
    ColumnProfiler, 
    CorrelationEngine, 
    DataValidator, 
    Sampler,
    ReportGenerator,
    profile_dataframe
)


class TestColumnProfiler:
    """Test ColumnProfiler class"""
    
    def setup_method(self):
        """Setup test data"""
        self.df = pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5, np.nan],
            'categorical': ['A', 'B', 'A', 'C', 'B', 'A'],
            'mixed': [1, 'A', 3, 'B', 5, np.nan],
            'all_null': [np.nan, np.nan, np.nan],
            'boolean': [True, False, True, False, True, False]
        })
        self.profiler = ColumnProfiler()
    
    def test_profile_basic(self):
        """Test basic profiling functionality"""
        profiles = self.profiler.profile(self.df)
        
        assert len(profiles) == 5
        
        # Check numeric column
        numeric_profile = next(p for p in profiles if p.name == 'numeric')
        assert numeric_profile.dtype == 'float64'
        assert numeric_profile.null_ratio == 1/6
        assert numeric_profile.unique_values == 5
        assert numeric_profile.value_range == (1.0, 5.0)
    
    def test_profile_categorical(self):
        """Test categorical column profiling"""
        profiles = self.profiler.profile(self.df)
        cat_profile = next(p for p in profiles if p.name == 'categorical')
        
        assert cat_profile.dtype == 'object'
        assert cat_profile.null_ratio == 0.0
        assert cat_profile.unique_values == 3
        assert cat_profile.clusters is not None
    
    def test_find_anomalies(self):
        """Test anomaly detection"""
        profiles = self.profiler.profile(self.df)
        anomalies = self.profiler.find_anomalies()
        
        # Should find high null ratio in all_null column
        null_anomalies = [a for a in anomalies if a['type'] == 'high_null_ratio']
        assert len(null_anomalies) > 0
    
    def test_get_summary(self):
        """Test summary generation"""
        profiles = self.profiler.profile(self.df)
        summary = self.profiler.get_summary()
        
        assert summary['total_columns'] == 5
        assert summary['numeric_columns'] >= 1
        assert summary['categorical_columns'] >= 1
        assert 'avg_null_ratio' in summary


class TestCorrelationEngine:
    """Test CorrelationEngine class"""
    
    def setup_method(self):
        """Setup test data"""
        np.random.seed(42)
        self.df = pd.DataFrame({
            'x': np.random.normal(0, 1, 100),
            'y': np.random.normal(0, 1, 100),
            'z': np.random.normal(0, 1, 100),
            'categorical': ['A', 'B'] * 50
        })
        self.engine = CorrelationEngine(self.df)
    
    def test_compute_correlation_matrix(self):
        """Test correlation matrix computation"""
        corr_matrix = self.engine.compute_correlation_matrix()
        
        assert corr_matrix.shape == (4, 4)  # All columns including encoded categorical
        assert not corr_matrix.isnull().any().any()
    
    def test_top_correlated_pairs(self):
        """Test finding top correlated pairs"""
        top_pairs = self.engine.top_correlated_pairs(threshold=0.1)
        
        assert isinstance(top_pairs, list)
        for pair in top_pairs:
            assert len(pair) == 3
            assert isinstance(pair[2], (int, float))
    
    def test_compute_multivariate_components(self):
        """Test PCA computation"""
        pc_df = self.engine.compute_multivariate_components(n_components=2)
        
        assert not pc_df.empty
        assert 'PC1' in pc_df.columns
        assert 'PC2' in pc_df.columns
        assert hasattr(self.engine, 'pca_info')
    
    def test_detect_collinearity(self):
        """Test collinearity detection"""
        collinear = self.engine.detect_collinearity(threshold=0.9)
        
        assert isinstance(collinear, list)
    
    def test_generate_report(self):
        """Test report generation"""
        report = self.engine.generate_report()
        
        assert 'bivariate_correlations' in report
        assert 'top_correlations' in report
        assert 'multivariate_analysis' in report
        assert 'summary' in report


class TestDataValidator:
    """Test DataValidator class"""
    
    def setup_method(self):
        """Setup test data"""
        self.df = pd.DataFrame({
            'age': [25, 30, 35, 40, 45, 150],  # One invalid age
            'email': ['test@example.com', 'invalid-email', 'user@domain.com', 'bad@', 'good@test.com', 'test@example.com'],
            'score': [85, 92, 78, 95, 88, 91],
            'category': ['A', 'B', 'A', 'C', 'B', 'A']
        })
        
        self.rules = {
            'age': {'min': 18, 'max': 100},
            'email': {'pattern': r'^[^@]+@[^@]+\.[^@]+$'},
            'score': {'min': 0, 'max': 100},
            'category': {'allowed_values': ['A', 'B', 'C']}
        }
        
        self.validator = DataValidator(self.df, self.rules)
    
    def test_validate_rules(self):
        """Test rule validation"""
        violations = self.validator.validate_rules()
        
        # Should find violations
        assert len(violations) > 0
        
        # Check age violations
        if 'age' in violations:
            age_violations = violations['age']
            assert any(v['type'] == 'above_max' for v in age_violations)
        
        # Check email violations
        if 'email' in violations:
            email_violations = violations['email']
            assert any(v['type'] == 'pattern_mismatch' for v in email_violations)
    
    def test_detect_outliers(self):
        """Test outlier detection"""
        outliers = self.validator.detect_outliers(method='iqr')
        
        assert isinstance(outliers, dict)
    
    def test_check_data_quality_metrics(self):
        """Test quality metrics calculation"""
        metrics = self.validator.check_data_quality_metrics()
        
        assert len(metrics) == 4  # One for each column
        for col_metrics in metrics.values():
            assert 'completeness' in col_metrics
            assert 'uniqueness' in col_metrics
            assert 'consistency' in col_metrics
            assert 'validity' in col_metrics
    
    def test_generate_validation_report(self):
        """Test validation report generation"""
        report = self.validator.generate_validation_report()
        
        assert 'custom_rule_violations' in report
        assert 'quality_metrics' in report
        assert 'summary' in report
    
    def test_suggest_improvements(self):
        """Test improvement suggestions"""
        suggestions = self.validator.suggest_improvements()
        
        assert isinstance(suggestions, list)


class TestSampler:
    """Test Sampler class"""
    
    def setup_method(self):
        """Setup test data"""
        self.df = pd.DataFrame({
            'x': range(1000),
            'y': range(1000),
            'category': ['A', 'B', 'C'] * 333 + ['A']
        })
        self.sampler = Sampler(sample_size=100)
    
    def test_sample_dataframe_random(self):
        """Test random sampling"""
        sampled = self.sampler.sample_dataframe(self.df, 'random')
        
        assert len(sampled) == 100
        assert list(sampled.columns) == list(self.df.columns)
    
    def test_sample_dataframe_stratified(self):
        """Test stratified sampling"""
        sampled = self.sampler.sample_dataframe(self.df, 'stratified')
        
        assert len(sampled) <= 100
        assert list(sampled.columns) == list(self.df.columns)
    
    def test_sample_dataframe_systematic(self):
        """Test systematic sampling"""
        sampled = self.sampler.sample_dataframe(self.df, 'systematic')
        
        assert len(sampled) == 100
        assert list(sampled.columns) == list(self.df.columns)


class TestReportGenerator:
    """Test ReportGenerator class"""
    
    def setup_method(self):
        """Setup test data"""
        self.generator = ReportGenerator()
        
        # Mock data
        self.context = {
            'table_name': 'test_table',
            'column_profiles': [
                {
                    'name': 'col1',
                    'dtype': 'int64',
                    'null_ratio': 0.1,
                    'unique_values': 100,
                    'value_range': (0, 100),
                    'statistics': {'mean': 50.0, 'std': 25.0}
                }
            ],
            'sample_size': 1000,
            'summary': {
                'total_columns': 1,
                'numeric_columns': 1,
                'categorical_columns': 0,
                'avg_null_ratio': 0.1
            }
        }
    
    def test_generate_html_report(self):
        """Test HTML report generation"""
        html_content = self.generator.generate_html_report(self.context)
        
        assert isinstance(html_content, str)
        assert '<html' in html_content
        assert 'test_table' in html_content
    
    def test_generate_markdown_report(self):
        """Test markdown report generation"""
        md_content = self.generator.generate_markdown_report(self.context)
        
        assert isinstance(md_content, str)
        assert '# Data Profiling Report' in md_content
        assert 'test_table' in md_content
    
    def test_generate_json_report(self):
        """Test JSON report generation"""
        json_content = self.generator.generate_json_report(self.context)
        
        assert isinstance(json_content, str)
        # Should be valid JSON
        import json
        parsed = json.loads(json_content)
        assert parsed['table_name'] == 'test_table'
    
    def test_generate_comprehensive_report(self):
        """Test comprehensive report generation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            reports = self.generator.generate_comprehensive_report(
                table_name='test_table',
                column_profiles=[Mock(to_dict=lambda: self.context['column_profiles'][0])],
                output_dir=temp_dir
            )
            
            assert 'html' in reports
            assert 'markdown' in reports
            assert 'json' in reports
            
            # Check files exist
            for report_path in reports.values():
                assert os.path.exists(report_path)


class TestIntegration:
    """Integration tests"""
    
    def test_profile_dataframe_integration(self):
        """Test complete DataFrame profiling pipeline"""
        # Create test data
        df = pd.DataFrame({
            'id': range(100),
            'value': np.random.normal(0, 1, 100),
            'category': ['A', 'B'] * 50
        })
        
        # Define validation rules
        rules = {
            'id': {'unique': True, 'not_null': True},
            'value': {'min': -3, 'max': 3}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run profiling
            results = profile_dataframe(
                df=df,
                table_name='test_integration',
                validation_rules=rules,
                output_dir=temp_dir
            )
            
            # Check results structure
            assert 'sample_size' in results
            assert 'column_profiles' in results
            assert 'correlation_results' in results
            assert 'validation_results' in results
            assert 'anomalies' in results
            assert 'recommendations' in results
            assert 'report_paths' in results
            
            # Check report files exist
            for report_path in results['report_paths'].values():
                assert os.path.exists(report_path)
    
    def test_error_handling(self):
        """Test error handling"""
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        
        with pytest.raises(Exception):
            profile_dataframe(empty_df, 'empty_table')
        
        # Test with invalid validation rules
        df = pd.DataFrame({'col': [1, 2, 3]})
        invalid_rules = {'nonexistent_col': {'min': 0}}
        
        # Should not raise exception, just log warnings
        results = profile_dataframe(df, 'test', invalid_rules)
        assert results is not None


if __name__ == "__main__":
    pytest.main([__file__]) 