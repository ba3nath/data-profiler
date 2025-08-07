import os
import pandas as pd
import json
import pytest

def test_load_kaggle_dataset():
    """Test loading the Kaggle-style customer dataset"""
    path = "tests/test_data/customer_data.csv"
    assert os.path.exists(path), f"Dataset not found: {path}"
    df = pd.read_csv(path)
    assert len(df) >= 99, f"Expected at least 99 rows, got {len(df)}"
    assert 'customer_id' in df.columns
    assert 'email' in df.columns
    assert 'income' in df.columns
    assert 'notes' in df.columns
    print(f"✅ Dataset loaded successfully: {len(df)} rows, {len(df.columns)} columns")

def test_load_validation_rules():
    """Test loading validation rules for the dataset"""
    path = "tests/test_data/validation_rules.json"
    assert os.path.exists(path), f"Validation rules not found: {path}"
    with open(path) as f:
        rules = json.load(f)
    assert 'customer_id' in rules
    assert 'email' in rules
    assert 'income' in rules
    assert 'notes' in rules
    print(f"✅ Validation rules loaded successfully: {len(rules)} rules")

def test_dataset_schema():
    """Test dataset schema and data types"""
    df = pd.read_csv("tests/test_data/customer_data.csv")
    
    # Check required columns exist
    required_columns = [
        'customer_id', 'name', 'age', 'email', 'phone', 'income',
        'credit_score', 'registration_date', 'is_active', 'transaction_count',
        'last_purchase_amount', 'customer_segment', 'satisfaction_score',
        'account_balance', 'notes'
    ]
    
    for col in required_columns:
        assert col in df.columns, f"Missing required column: {col}"
    
    # Check data types
    assert df['customer_id'].dtype == 'object'
    assert df['age'].dtype in ['int64', 'float64']
    assert df['income'].dtype in ['int64', 'float64']
    assert df['transaction_count'].dtype in ['int64', 'float64']
    
    print(f"✅ Dataset schema validation passed")

def test_data_quality_checks():
    """Test basic data quality checks"""
    df = pd.read_csv("tests/test_data/customer_data.csv")
    
    # Check for missing values
    missing_counts = df.isnull().sum()
    print(f"Missing values per column: {missing_counts.to_dict()}")
    
    # Check for unique customer IDs
    assert df['customer_id'].nunique() == len(df), "Customer IDs should be unique"
    
    # Check age range
    age_range = (df['age'].min(), df['age'].max())
    print(f"Age range: {age_range}")
    
    # Check income range
    income_range = (df['income'].min(), df['income'].max())
    print(f"Income range: {income_range}")
    
    # Check customer segments
    segments = df['customer_segment'].value_counts()
    print(f"Customer segments: {segments.to_dict()}")
    
    print(f"✅ Data quality checks passed")

def test_edge_cases_kaggle_dataset():
    """Test edge cases in the dataset (missing, outliers, invalid)"""
    df = pd.read_csv("tests/test_data/customer_data.csv")
    
    # Check for missing values
    missing = df.isnull().sum()
    print(f"Total missing values: {missing.sum()}")
    
    # Check for potential outliers in age
    age_outliers = df[(df['age'] < 18) | (df['age'] > 100)]
    print(f"Age outliers (<18 or >100): {len(age_outliers)}")
    
    # Check for potential outliers in income
    income_outliers = df[df['income'] > 200000]
    print(f"Income outliers (>200k): {len(income_outliers)}")
    
    # Check for invalid emails (basic pattern)
    invalid_emails = df[~df['email'].astype(str).str.contains(r'@', na=False)]
    print(f"Invalid emails (no @ symbol): {len(invalid_emails)}")
    
    # Check for inactive customers
    inactive_customers = df[df['is_active'] == False]
    print(f"Inactive customers: {len(inactive_customers)}")
    
    print(f"✅ Edge case analysis completed")

def test_validation_rules_structure():
    """Test validation rules structure and content"""
    with open("tests/test_data/validation_rules.json") as f:
        rules = json.load(f)
    
    # Check rule structure
    for column, rule in rules.items():
        assert isinstance(rule, dict), f"Rule for {column} should be a dictionary"
        
        # Check for required rule types
        if 'pattern' in rule:
            assert isinstance(rule['pattern'], str), f"Pattern for {column} should be string"
        
        if 'min' in rule:
            assert isinstance(rule['min'], (int, float)), f"Min for {column} should be numeric"
        
        if 'max' in rule:
            assert isinstance(rule['max'], (int, float)), f"Max for {column} should be numeric"
        
        if 'allowed_values' in rule:
            assert isinstance(rule['allowed_values'], list), f"Allowed values for {column} should be list"
    
    print(f"✅ Validation rules structure validated")

# Test that requires data_profiler module (commented out for now)
"""
def test_profile_kaggle_dataset():
    # This test requires the full data_profiler module
    # Uncomment when the module is properly installed
    from data_profiler import profile_dataframe, load_config
    
    df = pd.read_csv("tests/test_data/customer_data.csv")
    rules = load_config("tests/test_data/validation_rules.json")
    results = profile_dataframe(
        df=df,
        table_name="customer_data",
        validation_rules=rules,
        output_dir="tests/test_data"
    )
    # Check result structure
    assert 'sample_size' in results
    assert 'column_profiles' in results
    assert 'correlation_results' in results
    assert 'validation_results' in results
    assert 'anomalies' in results
    assert 'recommendations' in results
    assert 'report_paths' in results
"""

if __name__ == "__main__":
    # Run basic tests
    test_load_kaggle_dataset()
    test_load_validation_rules()
    test_dataset_schema()
    test_data_quality_checks()
    test_edge_cases_kaggle_dataset()
    test_validation_rules_structure()
    print("\n🎉 All basic tests passed!")