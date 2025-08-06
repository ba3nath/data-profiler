#!/usr/bin/env python3
"""
Basic test script to verify project structure
"""

import os
import sys

def test_project_structure():
    """Test that all required files and directories exist"""
    print("Testing project structure...")
    
    # Check main directories
    required_dirs = [
        'data_profiler',
        'data_profiler/connectors',
        'templates',
        'examples',
        'tests'
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} - MISSING")
            return False
    
    # Check main files
    required_files = [
        'data_profiler/__init__.py',
        'data_profiler/connectors/__init__.py',
        'data_profiler/connectors/base.py',
        'data_profiler/connectors/aws.py',
        'data_profiler/connectors/gcp.py',
        'data_profiler/connectors/azure.py',
        'data_profiler/sampler.py',
        'data_profiler/column_profiler.py',
        'data_profiler/correlation_engine.py',
        'data_profiler/data_validator.py',
        'data_profiler/report_generator.py',
        'data_profiler/openmetadata_integration.py',
        'data_profiler/orchestrator.py',
        'templates/report.html',
        'examples/demo.py',
        'examples/config_aws_redshift.json',
        'examples/config_gcp_bigquery.json',
        'tests/test_data_profiler.py',
        'requirements.txt',
        'README.md',
        'setup.py',
        'cli.py',
        'QUICK_START.md'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            return False
    
    return True

def test_imports():
    """Test that basic imports work (without external dependencies)"""
    print("\nTesting basic imports...")
    
    try:
        # Test importing the main package
        sys.path.insert(0, os.getcwd())
        
        # Test basic module imports
        import data_profiler
        print("✓ data_profiler package imported")
        
        # Test connector imports
        from data_profiler.connectors import CloudConnector, AWSConnector, GCPConnector, AzureConnector
        print("✓ Connector classes imported")
        
        # Test other module imports
        from data_profiler.sampler import Sampler
        from data_profiler.column_profiler import ColumnProfiler
        from data_profiler.correlation_engine import CorrelationEngine
        from data_profiler.data_validator import DataValidator
        from data_profiler.report_generator import ReportGenerator
        from data_profiler.openmetadata_integration import OpenMetadataIntegration
        from data_profiler.orchestrator import orchestrate, profile_dataframe
        print("✓ All main modules imported")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_class_instantiation():
    """Test that classes can be instantiated (without external dependencies)"""
    print("\nTesting class instantiation...")
    
    try:
        # Test basic classes that don't require external dependencies
        from data_profiler.sampler import Sampler
        from data_profiler.report_generator import ReportGenerator
        
        sampler = Sampler(sample_size=1000)
        print("✓ Sampler instantiated")
        
        generator = ReportGenerator()
        print("✓ ReportGenerator instantiated")
        
        return True
        
    except Exception as e:
        print(f"✗ Instantiation error: {e}")
        return False

def main():
    """Run all tests"""
    print("Data Profiler - Basic Structure Test")
    print("=" * 40)
    
    # Test project structure
    structure_ok = test_project_structure()
    
    # Test imports
    imports_ok = test_imports()
    
    # Test class instantiation
    instantiation_ok = test_class_instantiation()
    
    print("\n" + "=" * 40)
    print("Test Results:")
    print(f"Project Structure: {'✓ PASS' if structure_ok else '✗ FAIL'}")
    print(f"Basic Imports: {'✓ PASS' if imports_ok else '✗ FAIL'}")
    print(f"Class Instantiation: {'✓ PASS' if instantiation_ok else '✗ FAIL'}")
    
    if all([structure_ok, imports_ok, instantiation_ok]):
        print("\n🎉 All basic tests passed! The project structure is correct.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run the demo: python examples/demo.py")
        print("3. Check the CLI: python cli.py --help")
        return True
    else:
        print("\n❌ Some tests failed. Please check the missing files/modules.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 