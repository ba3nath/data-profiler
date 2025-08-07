#!/usr/bin/env python3
"""
Generate comprehensive test dataset for data profiling
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json
import os

def generate_test_dataset():
    """Generate a comprehensive test dataset with various data types and scenarios"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Generate 500 rows of data
    n_rows = 500
    
    # 1. Customer ID (unique identifier)
    customer_ids = [f"CUST_{i:04d}" for i in range(1, n_rows + 1)]
    
    # 2. Names (with some missing values)
    first_names = ["John", "Jane", "Mike", "Sarah", "David", "Lisa", "Tom", "Emma", "Alex", "Maria"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"]
    
    names = []
    for i in range(n_rows):
        if random.random() < 0.95:  # 5% missing names
            name = f"{random.choice(first_names)} {random.choice(last_names)}"
        else:
            name = np.nan
        names.append(name)
    
    # 3. Ages (with outliers and missing values)
    ages = []
    for i in range(n_rows):
        if random.random() < 0.92:  # 8% missing ages
            if random.random() < 0.95:  # 95% normal ages
                age = random.randint(18, 75)
            else:  # 5% outliers
                age = random.choice([0, 150, 200, -5])
        else:
            age = np.nan
        ages.append(age)
    
    # 4. Email addresses (with invalid formats)
    domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "company.com"]
    emails = []
    for i in range(n_rows):
        if random.random() < 0.90:  # 10% missing emails
            if random.random() < 0.85:  # 85% valid emails
                email = f"user{i}@{random.choice(domains)}"
            else:  # 15% invalid emails
                email = random.choice([
                    "invalid-email",
                    "user@",
                    "@domain.com",
                    "user@domain",
                    "user domain.com"
                ])
        else:
            email = np.nan
        emails.append(email)
    
    # 5. Phone numbers (with various formats)
    phone_formats = [
        "(555) 123-4567",
        "555-123-4567", 
        "555.123.4567",
        "5551234567",
        "+1-555-123-4567"
    ]
    phones = []
    for i in range(n_rows):
        if random.random() < 0.88:  # 12% missing phones
            if random.random() < 0.80:  # 80% valid phones
                phone = random.choice(phone_formats)
            else:  # 20% invalid phones
                phone = random.choice([
                    "123",
                    "abc-def-ghij",
                    "555-123",
                    "invalid"
                ])
        else:
            phone = np.nan
        phones.append(phone)
    
    # 6. Income (numeric with outliers)
    incomes = []
    for i in range(n_rows):
        if random.random() < 0.85:  # 15% missing income
            if random.random() < 0.90:  # 90% normal income
                income = random.randint(25000, 150000)
            else:  # 10% outliers
                income = random.choice([-5000, 0, 1000000, 9999999])
        else:
            income = np.nan
        incomes.append(income)
    
    # 7. Credit Score (categorical with numeric values)
    credit_scores = []
    for i in range(n_rows):
        if random.random() < 0.90:  # 10% missing credit scores
            score = random.choice(["Excellent", "Good", "Fair", "Poor", "Very Poor"])
        else:
            score = np.nan
        credit_scores.append(score)
    
    # 8. Registration Date (datetime with some invalid dates)
    base_date = datetime(2020, 1, 1)
    dates = []
    for i in range(n_rows):
        if random.random() < 0.95:  # 5% missing dates
            if random.random() < 0.90:  # 90% valid dates
                days_offset = random.randint(0, 1000)
                date = base_date + timedelta(days=days_offset)
                date_str = date.strftime("%Y-%m-%d")
            else:  # 10% invalid dates
                date_str = random.choice([
                    "2020-13-01",  # Invalid month
                    "2020-02-30",  # Invalid day
                    "invalid-date",
                    "2020/01/01",  # Wrong format
                    "01-01-2020"   # Wrong format
                ])
        else:
            date_str = np.nan
        dates.append(date_str)
    
    # 9. Is Active (boolean with some non-boolean values)
    active_status = []
    for i in range(n_rows):
        if random.random() < 0.92:  # 8% missing values
            if random.random() < 0.85:  # 85% valid boolean
                status = random.choice([True, False])
            else:  # 15% non-boolean values
                status = random.choice(["Yes", "No", "Y", "N", "1", "0", "Active", "Inactive"])
        else:
            status = np.nan
        active_status.append(status)
    
    # 10. Transaction Count (integer with some non-integer values)
    transaction_counts = []
    for i in range(n_rows):
        if random.random() < 0.88:  # 12% missing values
            if random.random() < 0.90:  # 90% valid integers
                count = random.randint(0, 1000)
            else:  # 10% non-integer values
                count = random.choice(["many", "few", "none", "unknown", 3.5, -1])
        else:
            count = np.nan
        transaction_counts.append(count)
    
    # 11. Last Purchase Amount (float with outliers)
    purchase_amounts = []
    for i in range(n_rows):
        if random.random() < 0.85:  # 15% missing values
            if random.random() < 0.85:  # 85% normal amounts
                amount = round(random.uniform(10.0, 500.0), 2)
            else:  # 15% outliers
                amount = random.choice([-50.0, 0.0, 99999.99, 1000000.0])
        else:
            amount = np.nan
        purchase_amounts.append(amount)
    
    # 12. Customer Segment (categorical with some typos)
    segments = ["Premium", "Gold", "Silver", "Bronze", "Basic"]
    customer_segments = []
    for i in range(n_rows):
        if random.random() < 0.90:  # 10% missing values
            if random.random() < 0.85:  # 85% valid segments
                segment = random.choice(segments)
            else:  # 15% typos or invalid values
                segment = random.choice([
                    "premium",  # lowercase
                    "GOLD",     # uppercase
                    "Silver ",  # trailing space
                    "Bronze.",  # trailing period
                    "Invalid"   # invalid value
                ])
        else:
            segment = np.nan
        customer_segments.append(segment)
    
    # 13. Satisfaction Score (1-10 scale with invalid values)
    satisfaction_scores = []
    for i in range(n_rows):
        if random.random() < 0.88:  # 12% missing values
            if random.random() < 0.80:  # 80% valid scores (1-10)
                score = random.randint(1, 10)
            else:  # 20% invalid scores
                score = random.choice([0, 11, 15, -1, "high", "low", "good"])
        else:
            score = np.nan
        satisfaction_scores.append(score)
    
    # 14. Account Balance (with currency symbols and formatting)
    balances = []
    for i in range(n_rows):
        if random.random() < 0.85:  # 15% missing values
            if random.random() < 0.75:  # 75% clean numeric values
                balance = round(random.uniform(-1000.0, 50000.0), 2)
            else:  # 25% formatted values
                balance = random.choice([
                    "$1,234.56",
                    "€2,345.67",
                    "£3,456.78",
                    "4,567.89",
                    "5,678.90 USD",
                    "invalid"
                ])
        else:
            balance = np.nan
        balances.append(balance)
    
    # 15. Notes (text with various lengths and special characters)
    notes = []
    for i in range(n_rows):
        if random.random() < 0.70:  # 30% missing notes
            if random.random() < 0.80:  # 80% normal notes
                note_length = random.randint(10, 100)
                note = f"Customer note #{i+1}: " + "".join(random.choices("abcdefghijklmnopqrstuvwxyz ", k=note_length))
            else:  # 20% special characters or very long notes
                note = random.choice([
                    "Customer with special chars: @#$%^&*()",
                    "Very long note: " + "x" * 500,
                    "Note with\nnewlines\nand\ttabs",
                    "Note with unicode: café, naïve, résumé",
                    "Empty note: "
                ])
        else:
            note = np.nan
        notes.append(note)
    
    # Create DataFrame
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'name': names,
        'age': ages,
        'email': emails,
        'phone': phones,
        'income': incomes,
        'credit_score': credit_scores,
        'registration_date': dates,
        'is_active': active_status,
        'transaction_count': transaction_counts,
        'last_purchase_amount': purchase_amounts,
        'customer_segment': customer_segments,
        'satisfaction_score': satisfaction_scores,
        'account_balance': balances,
        'notes': notes
    })
    
    return df

def generate_validation_rules():
    """Generate validation rules for the test dataset"""
    
    rules = {
        'customer_id': {
            'unique': True,
            'not_null': True,
            'pattern': r'^CUST_\d{4}$'
        },
        'name': {
            'not_null': False,  # Allow nulls
            'pattern': r'^[A-Z][a-z]+ [A-Z][a-z]+$'
        },
        'age': {
            'min': 18,
            'max': 100,
            'dtype': 'numeric'
        },
        'email': {
            'pattern': r'^[^@]+@[^@]+\.[^@]+$',
            'not_null': False
        },
        'phone': {
            'pattern': r'^[\d\-\(\)\.\+ ]+$',
            'not_null': False
        },
        'income': {
            'min': 0,
            'max': 1000000,
            'dtype': 'numeric'
        },
        'credit_score': {
            'allowed_values': ['Excellent', 'Good', 'Fair', 'Poor', 'Very Poor'],
            'not_null': False
        },
        'registration_date': {
            'pattern': r'^\d{4}-\d{2}-\d{2}$',
            'not_null': False
        },
        'is_active': {
            'allowed_values': [True, False],
            'not_null': False
        },
        'transaction_count': {
            'min': 0,
            'max': 10000,
            'dtype': 'integer'
        },
        'last_purchase_amount': {
            'min': 0,
            'max': 100000,
            'dtype': 'numeric'
        },
        'customer_segment': {
            'allowed_values': ['Premium', 'Gold', 'Silver', 'Bronze', 'Basic'],
            'not_null': False
        },
        'satisfaction_score': {
            'min': 1,
            'max': 10,
            'dtype': 'integer'
        },
        'account_balance': {
            'dtype': 'numeric',
            'not_null': False
        },
        'notes': {
            'max_length': 1000,
            'not_null': False
        }
    }
    
    return rules

def main():
    """Generate and save the test dataset"""
    
    print("Generating comprehensive test dataset...")
    
    # Generate dataset
    df = generate_test_dataset()
    
    # Generate validation rules
    rules = generate_validation_rules()
    
    # Save dataset
    dataset_path = "tests/test_data/customer_data.csv"
    df.to_csv(dataset_path, index=False)
    
    # Save validation rules
    rules_path = "tests/test_data/validation_rules.json"
    with open(rules_path, 'w') as f:
        json.dump(rules, f, indent=2)
    
    # Save dataset info
    info = {
        'rows': len(df),
        'columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.astype(str).to_dict(),
        'description': 'Comprehensive customer dataset for testing data profiling functionality'
    }
    
    info_path = "tests/test_data/dataset_info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"✅ Dataset generated successfully!")
    print(f"   📁 Dataset: {dataset_path}")
    print(f"   📁 Validation Rules: {rules_path}")
    print(f"   📁 Dataset Info: {info_path}")
    print(f"   📊 Rows: {len(df)}")
    print(f"   📊 Columns: {len(df.columns)}")
    print(f"   📊 Missing Values: {df.isnull().sum().sum()}")
    
    # Print sample statistics
    print(f"\n📈 Sample Statistics:")
    print(f"   - Age range: {df['age'].min()} to {df['age'].max()}")
    print(f"   - Income range: ${df['income'].min():,.0f} to ${df['income'].max():,.0f}")
    print(f"   - Unique customer segments: {df['customer_segment'].nunique()}")
    print(f"   - Email validity rate: {(df['email'].str.contains(r'^[^@]+@[^@]+\.[^@]+$', na=False).sum() / len(df) * 100):.1f}%")

if __name__ == "__main__":
    main() 