# Quick Start Guide

## Installation

```bash
# Install the package
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

## Basic Usage

### 1. Profile a CSV File

```python
from data_profiler import profile_dataframe
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# Profile the data
results = profile_dataframe(
    df=df,
    table_name="my_table",
    output_dir="reports"
)

print(f"Generated reports: {results['report_paths']}")
```

### 2. Profile Database Tables

```python
from data_profiler import orchestrate

# Configuration for AWS Redshift
config = {
    'cloud': 'aws',
    'db_type': 'redshift',
    'connection': {
        'host': 'your-cluster.redshift.amazonaws.com',
        'port': 5439,
        'database': 'your_database',
        'username': 'your_username',
        'password': 'your_password'
    },
    'sample_size': 10000,
    'tables': ['users', 'orders', 'products']
}

# Run profiling
results = orchestrate(config)
```

### 3. Using the Command Line

```bash
# Profile a CSV file
python cli.py profile-file data.csv --output-dir reports

# Profile database using config file
python cli.py profile-db config.json

# Run demo with sample data
python cli.py demo

# List supported formats
python cli.py formats
```

## Key Features

### Column Profiling
- Data type detection
- Null ratio analysis
- Value distribution analysis
- Statistical summaries (mean, std, percentiles)
- Cluster detection for numeric data

### Correlation Analysis
- Bivariate correlations (Pearson, Spearman)
- Multivariate analysis (PCA)
- Feature importance ranking
- Collinearity detection
- Feature clustering

### Data Validation
- Custom validation rules
- Great Expectations integration
- Data quality metrics
- Anomaly detection
- Improvement recommendations

### Report Generation
- HTML reports with interactive visualizations
- Markdown reports for documentation
- JSON reports for programmatic access
- Customizable templates

### Cloud Database Support
- AWS Redshift and RDS
- Google BigQuery
- Azure SQL Database and Synapse

## Configuration Examples

### AWS Redshift Configuration
```json
{
  "cloud": "aws",
  "db_type": "redshift",
  "connection": {
    "host": "your-cluster.redshift.amazonaws.com",
    "port": 5439,
    "database": "your_database",
    "username": "your_username",
    "password": "your_password"
  },
  "sample_size": 10000,
  "tables": ["users", "orders", "products"]
}
```

### Validation Rules
```json
{
  "age": {
    "min": 0,
    "max": 120
  },
  "email": {
    "pattern": "^[^@]+@[^@]+\\.[^@]+$"
  },
  "user_id": {
    "unique": true,
    "not_null": true
  }
}
```

## Output Structure

The profiler generates:
- **HTML Report**: Interactive web report with visualizations
- **Markdown Report**: Documentation-friendly format
- **JSON Report**: Machine-readable results
- **Summary Statistics**: Key metrics and insights

## Next Steps

1. **Explore the Examples**: Check the `examples/` directory for more detailed usage
2. **Customize Templates**: Modify HTML templates in `templates/` directory
3. **Add Validation Rules**: Define custom validation rules for your data
4. **Integrate with OpenMetadata**: Push results to your metadata store
5. **Run Tests**: Execute `pytest tests/` to verify functionality

## Support

- Check the main README.md for detailed documentation
- Review the test files for usage examples
- Open an issue for bugs or feature requests 