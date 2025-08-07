import os
import pandas as pd
import json
import pytest
import tempfile
import shutil

def test_full_kaggle_dataset_profiling():
    """Test full profiling pipeline with the Kaggle dataset"""
    try:
        from data_profiler import profile_dataframe, load_config
        
        # Load dataset
        df = pd.read_csv("tests/test_data/customer_data.csv")
        print(f"✅ Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Load validation rules
        rules = load_config("tests/test_data/validation_rules.json")
        print(f"✅ Validation rules loaded: {len(rules)} rules")
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run profiling
            results = profile_dataframe(
                df=df,
                table_name="customer_data",
                validation_rules=rules,
                output_dir=temp_dir
            )
            
            # Check result structure
            assert 'sample_size' in results, "Missing sample_size in results"
            assert 'column_profiles' in results, "Missing column_profiles in results"
            assert 'correlation_results' in results, "Missing correlation_results in results"
            assert 'validation_results' in results, "Missing validation_results in results"
            assert 'anomalies' in results, "Missing anomalies in results"
            assert 'recommendations' in results, "Missing recommendations in results"
            assert 'report_paths' in results, "Missing report_paths in results"
            
            # Check profiling output
            assert results['sample_size'] == len(df), f"Sample size mismatch: {results['sample_size']} vs {len(df)}"
            assert isinstance(results['column_profiles'], list), "Column profiles should be a list"
            assert isinstance(results['correlation_results'], dict), "Correlation results should be a dict"
            assert isinstance(results['validation_results'], dict), "Validation results should be a dict"
            assert isinstance(results['anomalies'], list), "Anomalies should be a list"
            assert isinstance(results['recommendations'], list), "Recommendations should be a list"
            
            # Check report files
            report_paths = results['report_paths']
            for ext in ['html', 'markdown', 'json']:
                if ext in report_paths:
                    assert os.path.exists(report_paths[ext]), f"Report file missing: {report_paths[ext]}"
                    print(f"✅ {ext.upper()} report generated: {report_paths[ext]}")
            
            print(f"✅ Full profiling completed successfully!")
            print(f"   - Column profiles: {len(results['column_profiles'])}")
            print(f"   - Anomalies found: {len(results['anomalies'])}")
            print(f"   - Recommendations: {len(results['recommendations'])}")
            
    except ImportError as e:
        pytest.skip(f"data_profiler module not available: {e}")
    except Exception as e:
        pytest.fail(f"Profiling failed: {e}")

def test_kaggle_dataset_validation():
    """Test validation functionality with the Kaggle dataset"""
    try:
        from data_profiler import DataValidator
        
        # Load dataset
        df = pd.read_csv("tests/test_data/customer_data.csv")
        
        # Load validation rules
        with open("tests/test_data/validation_rules.json") as f:
            rules = json.load(f)
        
        # Create validator
        validator = DataValidator(df, rules)
        
        # Run validation
        violations = validator.validate_rules()
        quality_metrics = validator.check_data_quality_metrics()
        
        # Check validation results
        assert isinstance(violations, dict), "Violations should be a dictionary"
        assert isinstance(quality_metrics, dict), "Quality metrics should be a dictionary"
        
        print(f"✅ Validation completed successfully!")
        print(f"   - Validation violations: {len(violations)}")
        print(f"   - Quality metrics: {len(quality_metrics)}")
        
        # Print some validation details
        for column, violations_list in violations.items():
            if violations_list:
                print(f"   - {column}: {len(violations_list)} violations")
        
    except ImportError as e:
        pytest.skip(f"DataValidator not available: {e}")
    except Exception as e:
        pytest.fail(f"Validation failed: {e}")

def test_kaggle_dataset_correlation():
    """Test correlation analysis with the Kaggle dataset"""
    try:
        from data_profiler import CorrelationEngine
        
        # Load dataset
        df = pd.read_csv("tests/test_data/customer_data.csv")
        
        # Create correlation engine
        engine = CorrelationEngine(df)
        
        # Run correlation analysis
        corr_matrix = engine.compute_correlation_matrix()
        top_pairs = engine.top_correlated_pairs(threshold=0.1)
        
        # Check correlation results
        assert not corr_matrix.empty, "Correlation matrix should not be empty"
        assert isinstance(top_pairs, list), "Top pairs should be a list"
        
        print(f"✅ Correlation analysis completed successfully!")
        print(f"   - Correlation matrix shape: {corr_matrix.shape}")
        print(f"   - Top correlated pairs: {len(top_pairs)}")
        
        # Print some correlation details
        if top_pairs:
            print(f"   - Strongest correlation: {top_pairs[0] if len(top_pairs) > 0 else 'None'}")
        
    except ImportError as e:
        pytest.skip(f"CorrelationEngine not available: {e}")
    except Exception as e:
        pytest.fail(f"Correlation analysis failed: {e}")

def test_kaggle_dataset_column_profiling():
    """Test column profiling with the Kaggle dataset"""
    try:
        from data_profiler import ColumnProfiler
        
        # Load dataset
        df = pd.read_csv("tests/test_data/customer_data.csv")
        
        # Create profiler
        profiler = ColumnProfiler()
        
        # Run profiling
        profiles = profiler.profile(df)
        anomalies = profiler.find_anomalies()
        summary = profiler.get_summary()
        
        # Check profiling results
        assert isinstance(profiles, list), "Profiles should be a list"
        assert len(profiles) == len(df.columns), f"Expected {len(df.columns)} profiles, got {len(profiles)}"
        assert isinstance(anomalies, list), "Anomalies should be a list"
        assert isinstance(summary, dict), "Summary should be a dictionary"
        
        print(f"✅ Column profiling completed successfully!")
        print(f"   - Column profiles: {len(profiles)}")
        print(f"   - Anomalies found: {len(anomalies)}")
        print(f"   - Summary keys: {list(summary.keys())}")
        
        # Print some profiling details
        for profile in profiles[:3]:  # Show first 3 profiles
            print(f"   - {profile.name}: {profile.dtype}, null_ratio={profile.null_ratio:.2f}")
        
    except ImportError as e:
        pytest.skip(f"ColumnProfiler not available: {e}")
    except Exception as e:
        pytest.fail(f"Column profiling failed: {e}")

if __name__ == "__main__":
    # Run all tests
    print("🧪 Running Kaggle Dataset Full Tests...")
    print("=" * 50)
    
    test_full_kaggle_dataset_profiling()
    test_kaggle_dataset_validation()
    test_kaggle_dataset_correlation()
    test_kaggle_dataset_column_profiling()
    
    print("\n🎉 All Kaggle dataset tests completed!") 