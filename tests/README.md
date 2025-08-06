# Data Profiler Test Suite

This directory contains comprehensive test cases for the Data Profiler project.

## Test Structure

```
tests/
├── README.md                    # This file
├── test_data_profiler.py        # Core component tests
├── test_connectors.py           # Cloud connector tests
├── test_cli.py                  # CLI functionality tests
├── test_orchestrator.py         # Orchestrator and utility tests
└── conftest.py                  # Pytest configuration (if needed)
```

## Test Categories

### 1. Unit Tests (`test_data_profiler.py`)
- **ColumnProfiler**: Tests for data column profiling functionality
- **CorrelationEngine**: Tests for correlation analysis
- **DataValidator**: Tests for data validation and quality checks
- **Sampler**: Tests for data sampling strategies
- **ReportGenerator**: Tests for report generation

### 2. Connector Tests (`test_connectors.py`)
- **AWSConnector**: Tests for AWS Redshift/RDS connections
- **GCPConnector**: Tests for Google BigQuery connections
- **AzureConnector**: Tests for Azure SQL Database connections
- **Integration**: Tests for connector factory patterns

### 3. CLI Tests (`test_cli.py`)
- **Command Parsing**: Tests for CLI argument handling
- **File Processing**: Tests for different file formats (CSV, Excel, JSON, Parquet)
- **Database Profiling**: Tests for database profiling commands
- **Error Handling**: Tests for various error scenarios

### 4. Orchestrator Tests (`test_orchestrator.py`)
- **Orchestration**: Tests for main orchestration pipeline
- **Configuration**: Tests for config loading and validation
- **Utility Functions**: Tests for helper functions
- **Integration**: Tests for end-to-end workflows

## Running Tests

### Quick Start
```bash
# Run all tests
python run_tests.py

# Run with coverage
python run_tests.py --coverage

# Run only unit tests
python run_tests.py --unit

# Run only integration tests
python run_tests.py --integration

# Skip slow tests
python run_tests.py --fast

# Run in parallel
python run_tests.py --parallel
```

### Using pytest directly
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_connectors.py

# Run specific test class
pytest tests/test_data_profiler.py::TestColumnProfiler

# Run specific test method
pytest tests/test_data_profiler.py::TestColumnProfiler::test_profile_basic

# Run with markers
pytest -m "not slow"
pytest -m integration
pytest -m unit
```

### Coverage Reports
```bash
# Generate coverage report
pytest --cov=data_profiler --cov-report=html

# View coverage in browser
open htmlcov/index.html
```

## Test Markers

The test suite uses pytest markers to categorize tests:

- `@pytest.mark.unit`: Unit tests (fast, isolated)
- `@pytest.mark.integration`: Integration tests (slower, require dependencies)
- `@pytest.mark.slow`: Slow tests (can be skipped with `--fast`)
- `@pytest.mark.cloud`: Tests requiring cloud connectivity
- `@pytest.mark.database`: Tests requiring database connectivity

## Mocking Strategy

The tests use extensive mocking to avoid external dependencies:

### Database Connections
```python
@patch('data_profiler.connectors.aws.create_engine')
def test_aws_connection(self, mock_create_engine):
    # Test AWS connector without real database
```

### File Operations
```python
with tempfile.NamedTemporaryFile() as f:
    # Test file operations with temporary files
```

### External APIs
```python
@patch('requests.get')
def test_api_call(self, mock_get):
    # Test API calls without real network requests
```

## Test Data

### Sample DataFrames
```python
self.test_df = pd.DataFrame({
    'id': range(100),
    'name': [f'user_{i}' for i in range(100)],
    'age': np.random.randint(18, 80, 100),
    'score': np.random.normal(70, 15, 100),
    'category': np.random.choice(['A', 'B', 'C'], 100)
})
```

### Configuration Examples
```python
self.test_config = {
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
```

## Best Practices

### 1. Test Isolation
- Each test should be independent
- Use `setup_method()` for test data
- Clean up resources in `teardown_method()`

### 2. Descriptive Names
```python
def test_profile_dataframe_with_validation_rules(self):
    """Test DataFrame profiling with custom validation rules"""
```

### 3. Assertion Messages
```python
assert len(profiles) == 5, f"Expected 5 profiles, got {len(profiles)}"
```

### 4. Error Testing
```python
with pytest.raises(ValueError, match="Invalid configuration"):
    function_under_test(invalid_input)
```

### 5. Edge Cases
- Empty DataFrames
- Missing configuration
- Network failures
- Invalid file formats

## Continuous Integration

The test suite is designed to run in CI/CD pipelines:

### GitHub Actions Example
```yaml
- name: Run Tests
  run: |
    pip install -r requirements-test.txt
    python run_tests.py --coverage --fast
```

### Pre-commit Hooks
```yaml
repos:
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest
        language: system
        pass_filenames: false
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements-test.txt
   ```

2. **Mock Issues**: Check import paths in `@patch` decorators
   ```python
   @patch('data_profiler.connectors.aws.create_engine')  # Correct
   @patch('connectors.aws.create_engine')               # Wrong
   ```

3. **File Permission Errors**: Ensure test runner is executable
   ```bash
   chmod +x run_tests.py
   ```

4. **Coverage Issues**: Check that source files are in the correct location
   ```bash
   pytest --cov=data_profiler --cov-report=term-missing
   ```

### Debug Mode
```bash
# Run with debug output
pytest -v -s

# Run single test with debug
pytest tests/test_data_profiler.py::TestColumnProfiler::test_profile_basic -v -s
```

## Performance

### Test Execution Times
- **Unit Tests**: < 1 second each
- **Integration Tests**: 1-5 seconds each
- **Slow Tests**: 5-30 seconds each

### Optimization Tips
- Use `@pytest.mark.slow` for time-consuming tests
- Mock external dependencies
- Use `--parallel` for faster execution
- Skip tests that don't apply to your changes

## Contributing

When adding new tests:

1. Follow the existing naming conventions
2. Add appropriate markers
3. Include docstrings
4. Test both success and failure cases
5. Update this documentation if needed

### Test Template
```python
class TestNewFeature:
    """Test new feature functionality"""
    
    def setup_method(self):
        """Setup test data"""
        pass
    
    def test_feature_success(self):
        """Test successful feature execution"""
        # Arrange
        # Act
        # Assert
    
    def test_feature_failure(self):
        """Test feature failure handling"""
        # Arrange
        # Act & Assert
        with pytest.raises(ExpectedException):
            function_under_test()
``` 