#!/usr/bin/env python3
"""
Demo script for Data Profiler
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from data_profiler import orchestrate, profile_dataframe


def create_sample_data():
    """Create sample data for demonstration"""
    np.random.seed(42)
    random.seed(42)
    
    # Generate sample data
    n_rows = 1000
    
    # User data
    user_ids = range(1, n_rows + 1)
    ages = np.random.normal(35, 15, n_rows).astype(int)
    ages = np.clip(ages, 18, 80)
    
    # Generate emails
    domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'company.com']
    emails = [f"user{i}@{random.choice(domains)}" for i in range(n_rows)]
    
    # Generate some invalid emails
    invalid_emails = ['invalid-email', 'no-at-sign', '@nodomain', 'spaces @email.com']
    for i in range(50):
        emails[random.randint(0, n_rows-1)] = random.choice(invalid_emails)
    
    # Generate purchase data
    purchase_amounts = np.random.exponential(100, n_rows)
    purchase_amounts = np.clip(purchase_amounts, 10, 1000)
    
    # Generate categories
    categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports']
    purchase_categories = [random.choice(categories) for _ in range(n_rows)]
    
    # Generate dates
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=random.randint(0, 365)) for _ in range(n_rows)]
    
    # Generate some null values
    ages[random.sample(range(n_rows), 50)] = np.nan
    purchase_amounts[random.sample(range(n_rows), 30)] = np.nan
    
    # Create DataFrame
    df = pd.DataFrame({
        'user_id': user_ids,
        'age': ages,
        'email': emails,
        'purchase_amount': purchase_amounts,
        'category': purchase_categories,
        'purchase_date': dates,
        'is_premium': [random.choice([True, False]) for _ in range(n_rows)],
        'rating': np.random.normal(4.0, 0.5, n_rows).round(1),
        'review_count': np.random.poisson(5, n_rows)
    })
    
    return df


def demo_dataframe_profiling():
    """Demonstrate DataFrame profiling"""
    print("=== DataFrame Profiling Demo ===\n")
    
    # Create sample data
    df = create_sample_data()
    print(f"Created sample dataset with {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}\n")
    
    # Define validation rules
    validation_rules = {
        'age': {
            'min': 18,
            'max': 80
        },
        'email': {
            'pattern': r'^[^@]+@[^@]+\.[^@]+$'
        },
        'purchase_amount': {
            'min': 0
        },
        'user_id': {
            'unique': True,
            'not_null': True
        },
        'rating': {
            'min': 1.0,
            'max': 5.0
        }
    }
    
    # Profile the DataFrame
    try:
        results = profile_dataframe(
            df=df,
            table_name="sample_customer_data",
            validation_rules=validation_rules,
            output_dir="demo_reports"
        )
        
        print("=== Profiling Results ===\n")
        
        # Print summary
        print(f"Sample size: {results['sample_size']}")
        print(f"Columns profiled: {len(results['column_profiles'])}")
        print(f"Anomalies found: {len(results['anomalies'])}")
        print(f"Recommendations: {len(results['recommendations'])}")
        
        # Print some anomalies
        if results['anomalies']:
            print("\n=== Data Quality Anomalies ===")
            for anomaly in results['anomalies'][:3]:  # Show first 3
                print(f"- {anomaly['type']}: {anomaly['description']}")
        
        # Print some recommendations
        if results['recommendations']:
            print("\n=== Recommendations ===")
            for rec in results['recommendations'][:3]:  # Show first 3
                print(f"- {rec}")
        
        # Print report paths
        print(f"\n=== Generated Reports ===")
        for report_type, path in results['report_paths'].items():
            print(f"- {report_type}: {path}")
        
        return results
        
    except Exception as e:
        print(f"Error during profiling: {e}")
        return None


def demo_correlation_analysis():
    """Demonstrate correlation analysis"""
    print("\n=== Correlation Analysis Demo ===\n")
    
    from data_profiler.correlation_engine import CorrelationEngine
    
    # Create sample data
    df = create_sample_data()
    
    # Initialize correlation engine
    engine = CorrelationEngine(df)
    
    # Get top correlations
    top_correlations = engine.top_correlated_pairs(threshold=0.3)
    
    print("Top Correlations (threshold: 0.3):")
    for col1, col2, corr in top_correlations[:5]:
        print(f"- {col1} ↔ {col2}: {corr:.3f}")
    
    # Get feature importance
    feature_importance = engine.get_feature_importance()
    
    print("\nFeature Importance (based on PCA):")
    for feature, importance in sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
        print(f"- {feature}: {importance:.3f}")
    
    # Suggest feature reduction
    suggested_removal = engine.suggest_feature_reduction()
    if suggested_removal:
        print(f"\nSuggested features for removal (high collinearity): {suggested_removal}")


def demo_validation():
    """Demonstrate data validation"""
    print("\n=== Data Validation Demo ===\n")
    
    from data_profiler.data_validator import DataValidator
    
    # Create sample data
    df = create_sample_data()
    
    # Define validation rules
    rules = {
        'age': {'min': 18, 'max': 80},
        'email': {'pattern': r'^[^@]+@[^@]+\.[^@]+$'},
        'purchase_amount': {'min': 0},
        'user_id': {'unique': True, 'not_null': True},
        'rating': {'min': 1.0, 'max': 5.0}
    }
    
    # Initialize validator
    validator = DataValidator(df, rules)
    
    # Run validation
    validation_results = validator.generate_validation_report()
    
    print("Validation Results:")
    print(f"- Total violations: {validation_results['summary']['total_violations']}")
    print(f"- Average quality score: {validation_results['summary']['avg_quality_score']:.2%}")
    print(f"- Columns with violations: {validation_results['summary']['columns_with_violations']}")
    
    # Show specific violations
    if validation_results['custom_rule_violations']:
        print("\nRule Violations:")
        for column, violations in validation_results['custom_rule_violations'].items():
            for violation in violations:
                print(f"- {column}: {violation['type']} ({violation['count']} violations)")


def main():
    """Main demo function"""
    print("Data Profiler Demo")
    print("=" * 50)
    
    # Demo 1: DataFrame profiling
    results = demo_dataframe_profiling()
    
    # Demo 2: Correlation analysis
    demo_correlation_analysis()
    
    # Demo 3: Data validation
    demo_validation()
    
    print("\n" + "=" * 50)
    print("Demo completed! Check the 'demo_reports' directory for generated reports.")
    print("You can open the HTML report in your browser to see the full analysis.")


if __name__ == "__main__":
    main() 