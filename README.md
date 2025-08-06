# Data Profiler

A modular system that connects to cloud databases (AWS, GCP, Azure), samples large datasets, profiles them, and generates comprehensive reports. It integrates with open-source tools like SQLAlchemy, OpenMetadata, YData Profiling, and Great Expectations to provide scalable and extensible data quality insights.

## Features

- **Cloud Connectors**: Support for AWS Redshift/RDS, Google BigQuery, and Azure SQL Database
- **Intelligent Sampling**: Efficient sampling strategies for large datasets
- **Column Profiling**: Detailed analysis of data types, null ratios, value distributions
- **Correlation Analysis**: Bivariate and multivariate correlation detection
- **Data Validation**: Integration with Great Expectations for data quality checks
- **Report Generation**: HTML and markdown reports with interactive visualizations
- **OpenMetadata Integration**: Push profiling results to metadata stores

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from data_profiler.orchestrator import orchestrate

config = {
    'cloud': 'aws',
    'connection': {
        'host': 'your-redshift-cluster.redshift.amazonaws.com',
        'port': 5439,
        'database': 'your_database',
        'username': 'your_username',
        'password': 'your_password'
    },
    'sample_size': 10000,
    'tables': ['table1', 'table2']
}

orchestrate(config)
```

## Project Structure

```
data-profiler/
├── data_profiler/
│   ├── __init__.py
│   ├── connectors/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── aws.py
│   │   ├── gcp.py
│   │   └── azure.py
│   ├── sampler.py
│   ├── column_profiler.py
│   ├── correlation_engine.py
│   ├── data_validator.py
│   ├── report_generator.py
│   ├── openmetadata_integration.py
│   └── orchestrator.py
├── templates/
│   └── report.html
├── requirements.txt
└── README.md
```

## Usage Examples

### Basic Profiling

```python
from data_profiler.column_profiler import ColumnProfiler
import pandas as pd

df = pd.read_csv('your_data.csv')
profiler = ColumnProfiler()
profiles = profiler.profile(df)

for profile in profiles:
    print(f"Column: {profile.name}")
    print(f"Type: {profile.dtype}")
    print(f"Null ratio: {profile.null_ratio:.2%}")
    print(f"Unique values: {profile.unique_values}")
```

### Correlation Analysis

```python
from data_profiler.correlation_engine import CorrelationEngine

engine = CorrelationEngine(df)
correlations = engine.compute_bivariate_correlations()
top_pairs = engine.top_correlated_pairs(threshold=0.7)
```

### Data Validation

```python
from data_profiler.data_validator import DataValidator

rules = {
    'age': {'min': 0, 'max': 120},
    'email': {'pattern': r'^[^@]+@[^@]+\.[^@]+$'}
}

validator = DataValidator(df, rules)
results = validator.report()
```

## Configuration

The system supports various configuration options for different cloud providers and sampling strategies. See the individual module documentation for detailed configuration examples.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License 