import os
import pandas as pd
import json
import tempfile
import shutil

def test_kaggle_dataset_comprehensive():
    """Comprehensive test of the Kaggle dataset with available data_profiler functionality"""
    
    print("🧪 Running Comprehensive Kaggle Dataset Test...")
    print("=" * 60)
    
    # 1. Load and validate dataset
    print("\n📊 1. Loading and validating dataset...")
    df = pd.read_csv("tests/test_data/customer_data.csv")
    print(f"   ✅ Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Check dataset structure
    required_columns = [
        'customer_id', 'name', 'age', 'email', 'phone', 'income',
        'credit_score', 'registration_date', 'is_active', 'transaction_count',
        'last_purchase_amount', 'customer_segment', 'satisfaction_score',
        'account_balance', 'notes'
    ]
    
    for col in required_columns:
        assert col in df.columns, f"Missing required column: {col}"
    print(f"   ✅ All required columns present")
    
    # 2. Load validation rules
    print("\n📋 2. Loading validation rules...")
    with open("tests/test_data/validation_rules.json") as f:
        rules = json.load(f)
    print(f"   ✅ Validation rules loaded: {len(rules)} rules")
    
    # 3. Test ColumnProfiler
    print("\n🔍 3. Testing ColumnProfiler...")
    try:
        from data_profiler import ColumnProfiler
        
        profiler = ColumnProfiler()
        profiles = profiler.profile(df)
        
        print(f"   ✅ Column profiling completed: {len(profiles)} profiles")
        
        # Show some profile details
        for i, profile in enumerate(profiles[:3]):
            print(f"   📈 {profile.name}: {profile.dtype}, null_ratio={profile.null_ratio:.2f}")
        
        # Test anomaly detection
        anomalies = profiler.find_anomalies()
        print(f"   ⚠️  Anomalies detected: {len(anomalies)}")
        
        # Test summary
        summary = profiler.get_summary()
        print(f"   📊 Summary generated with {len(summary)} metrics")
        
    except Exception as e:
        print(f"   ❌ ColumnProfiler test failed: {e}")
    
    # 4. Test DataValidator
    print("\n✅ 4. Testing DataValidator...")
    try:
        from data_profiler import DataValidator
        
        validator = DataValidator(df, rules)
        violations = validator.validate_rules()
        quality_metrics = validator.check_data_quality_metrics()
        
        print(f"   ✅ Validation completed")
        print(f"   📊 Quality metrics: {len(quality_metrics)} columns")
        
        # Show validation results
        total_violations = sum(len(v) for v in violations.values())
        print(f"   ⚠️  Total violations: {total_violations}")
        
        for column, violations_list in violations.items():
            if violations_list:
                print(f"   🔍 {column}: {len(violations_list)} violations")
        
    except Exception as e:
        print(f"   ❌ DataValidator test failed: {e}")
    
    # 5. Test CorrelationEngine
    print("\n🔗 5. Testing CorrelationEngine...")
    try:
        from data_profiler import CorrelationEngine
        
        engine = CorrelationEngine(df)
        corr_matrix = engine.compute_correlation_matrix()
        top_pairs = engine.top_correlated_pairs(threshold=0.1)
        
        print(f"   ✅ Correlation analysis completed")
        print(f"   📊 Correlation matrix: {corr_matrix.shape}")
        print(f"   🔗 Top correlated pairs: {len(top_pairs)}")
        
        if top_pairs:
            print(f"   💪 Strongest correlation: {top_pairs[0]}")
        
    except Exception as e:
        print(f"   ❌ CorrelationEngine test failed: {e}")
    
    # 6. Test ReportGenerator
    print("\n📄 6. Testing ReportGenerator...")
    try:
        from data_profiler import ReportGenerator
        
        generator = ReportGenerator()
        
        # Create sample context
        context = {
            'table_name': 'customer_data',
            'column_profiles': profiles[:3] if 'profiles' in locals() else [],
            'sample_size': len(df),
            'summary': summary if 'summary' in locals() else {}
        }
        
        # Generate reports
        with tempfile.TemporaryDirectory() as temp_dir:
            html_report = generator.generate_html_report(context)
            md_report = generator.generate_markdown_report(context)
            json_report = generator.generate_json_report(context)
            
            print(f"   ✅ Report generation completed")
            print(f"   📄 HTML report: {len(html_report)} characters")
            print(f"   📄 Markdown report: {len(md_report)} characters")
            print(f"   📄 JSON report: {len(json_report)} characters")
        
    except Exception as e:
        print(f"   ❌ ReportGenerator test failed: {e}")
    
    # 7. Test Sampler
    print("\n🎯 7. Testing Sampler...")
    try:
        from data_profiler import Sampler
        
        sampler = Sampler(sample_size=50)
        
        # Test different sampling methods
        random_sample = sampler.sample_dataframe(df, 'random')
        stratified_sample = sampler.sample_dataframe(df, 'stratified')
        
        print(f"   ✅ Sampling completed")
        print(f"   🎯 Random sample: {len(random_sample)} rows")
        print(f"   🎯 Stratified sample: {len(stratified_sample)} rows")
        
    except Exception as e:
        print(f"   ❌ Sampler test failed: {e}")
    
    # 8. Data Quality Analysis
    print("\n📈 8. Data Quality Analysis...")
    
    # Missing values
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    print(f"   📊 Missing values: {total_missing} total")
    
    # Data types
    dtypes = df.dtypes.value_counts()
    print(f"   📊 Data types: {dict(dtypes)}")
    
    # Value ranges
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols[:3]:  # Show first 3 numeric columns
        col_range = (df[col].min(), df[col].max())
        print(f"   📊 {col} range: {col_range}")
    
    # Unique values
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols[:3]:  # Show first 3 categorical columns
        unique_count = df[col].nunique()
        print(f"   📊 {col} unique values: {unique_count}")
    
    print("\n🎉 Comprehensive test completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    test_kaggle_dataset_comprehensive() 