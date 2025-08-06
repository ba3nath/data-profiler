"""
Main orchestrator for the data profiling pipeline
"""

import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

from .connectors import AWSConnector, GCPConnector, AzureConnector
from .sampler import Sampler
from .column_profiler import ColumnProfiler
from .correlation_engine import CorrelationEngine
from .data_validator import DataValidator
from .report_generator import ReportGenerator
from .openmetadata_integration import OpenMetadataIntegration


def get_connector(cloud_provider: str, config: Dict) -> Any:
    """
    Get appropriate cloud connector based on provider
    
    Args:
        cloud_provider: Cloud provider ('aws', 'gcp', 'azure')
        config: Connection configuration
        
    Returns:
        Cloud connector instance
    """
    if cloud_provider.lower() == 'aws':
        return AWSConnector(config)
    elif cloud_provider.lower() == 'gcp':
        return GCPConnector(config)
    elif cloud_provider.lower() == 'azure':
        return AzureConnector(config)
    else:
        raise ValueError(f"Unsupported cloud provider: {cloud_provider}")


def discover_tables(connector: Any, config: Dict) -> List[str]:
    """
    Discover tables in the database
    
    Args:
        connector: Cloud connector instance
        config: Configuration dictionary
        
    Returns:
        List of table names
    """
    try:
        # Use connector's get_tables method if available
        if hasattr(connector, 'get_tables'):
            return connector.get_tables()
        
        # Fallback: use specified tables from config
        return config.get('tables', [])
    except Exception as e:
        print(f"Error discovering tables: {e}")
        return config.get('tables', [])


def orchestrate(config: Dict) -> Dict[str, Any]:
    """
    Main orchestration function for data profiling pipeline
    
    Args:
        config: Configuration dictionary containing:
            - cloud: Cloud provider ('aws', 'gcp', 'azure')
            - connection: Database connection parameters
            - sample_size: Number of rows to sample
            - tables: List of tables to profile (optional)
            - validation_rules: Data validation rules (optional)
            - openmetadata: OpenMetadata configuration (optional)
            - output_dir: Output directory for reports (optional)
    
    Returns:
        Dictionary with profiling results and report paths
    """
    print("Starting data profiling pipeline...")
    
    # Extract configuration
    cloud_provider = config.get('cloud', 'aws')
    connection_config = config.get('connection', {})
    sample_size = config.get('sample_size', 10000)
    tables = config.get('tables', [])
    validation_rules = config.get('validation_rules', {})
    openmetadata_config = config.get('openmetadata', {})
    output_dir = config.get('output_dir', 'reports')
    
    # Initialize components
    try:
        # Get cloud connector
        connector = get_connector(cloud_provider, connection_config)
        print(f"Connected to {cloud_provider.upper()} database")
        
        # Test connection
        if not connector.test_connection():
            raise Exception("Database connection test failed")
        
        # Initialize components
        sampler = Sampler(sample_size=sample_size)
        column_profiler = ColumnProfiler()
        report_generator = ReportGenerator()
        
        # Initialize OpenMetadata integration if configured
        openmetadata_integration = None
        if openmetadata_config:
            try:
                openmetadata_integration = OpenMetadataIntegration(openmetadata_config)
                print("OpenMetadata integration initialized")
            except Exception as e:
                print(f"Warning: OpenMetadata integration failed: {e}")
        
        # Discover tables if not specified
        if not tables:
            tables = discover_tables(connector, config)
            print(f"Discovered {len(tables)} tables")
        
        if not tables:
            raise Exception("No tables found to profile")
        
        # Get database engine
        engine = connector.get_engine()
        
        # Process each table
        results = {}
        
        for table_name in tables:
            print(f"\nProcessing table: {table_name}")
            
            try:
                # Step 1: Sample data
                print("  Sampling data...")
                df_sample = sampler.sample_table(engine, table_name)
                print(f"  Sampled {len(df_sample)} rows")
                
                # Step 2: Profile columns
                print("  Profiling columns...")
                column_profiles = column_profiler.profile(df_sample)
                print(f"  Profiled {len(column_profiles)} columns")
                
                # Step 3: Correlation analysis
                print("  Analyzing correlations...")
                correlation_engine = CorrelationEngine(df_sample)
                correlation_results = correlation_engine.generate_report()
                
                # Step 4: Data validation
                print("  Validating data...")
                validator = DataValidator(df_sample, validation_rules)
                validation_results = validator.generate_validation_report()
                
                # Step 5: Generate reports
                print("  Generating reports...")
                anomalies = column_profiler.find_anomalies()
                recommendations = validator.suggest_improvements()
                
                # Generate comprehensive report
                report_paths = report_generator.generate_comprehensive_report(
                    table_name=table_name,
                    column_profiles=column_profiles,
                    correlations=correlation_results,
                    validation_results=validation_results,
                    anomalies=anomalies,
                    recommendations=recommendations,
                    sample_size=len(df_sample),
                    output_dir=output_dir
                )
                
                # Step 6: Push to OpenMetadata if configured
                openmetadata_results = {}
                if openmetadata_integration:
                    print("  Pushing to OpenMetadata...")
                    openmetadata_results = openmetadata_integration.push_comprehensive_results(
                        table_name=table_name,
                        database_name=connection_config.get('database', 'unknown'),
                        schema_name=connection_config.get('schema', 'public'),
                        column_profiles=[p.to_dict() for p in column_profiles],
                        correlation_results=correlation_results,
                        validation_results=validation_results,
                        table_stats={'row_count': len(df_sample), 'column_count': len(df_sample.columns)},
                        sample_size=len(df_sample)
                    )
                
                # Store results
                results[table_name] = {
                    'sample_size': len(df_sample),
                    'column_profiles': [p.to_dict() for p in column_profiles],
                    'correlation_results': correlation_results,
                    'validation_results': validation_results,
                    'anomalies': anomalies,
                    'recommendations': recommendations,
                    'report_paths': report_paths,
                    'openmetadata_results': openmetadata_results,
                    'processing_time': datetime.now().isoformat()
                }
                
                print(f"  Completed processing {table_name}")
                
            except Exception as e:
                print(f"  Error processing table {table_name}: {e}")
                results[table_name] = {
                    'error': str(e),
                    'processing_time': datetime.now().isoformat()
                }
        
        # Generate summary report
        summary = generate_summary_report(results)
        
        print(f"\nPipeline completed. Processed {len(tables)} tables.")
        print(f"Reports saved to: {output_dir}")
        
        return {
            'summary': summary,
            'results': results,
            'config': config,
            'completion_time': datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise


def generate_summary_report(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate summary report from all table results
    
    Args:
        results: Dictionary with results for each table
        
    Returns:
        Summary statistics
    """
    summary = {
        'total_tables': len(results),
        'successful_tables': 0,
        'failed_tables': 0,
        'total_rows_sampled': 0,
        'total_columns': 0,
        'total_anomalies': 0,
        'total_recommendations': 0,
        'avg_quality_score': 0.0
    }
    
    quality_scores = []
    
    for table_name, result in results.items():
        if 'error' in result:
            summary['failed_tables'] += 1
            continue
        
        summary['successful_tables'] += 1
        summary['total_rows_sampled'] += result.get('sample_size', 0)
        summary['total_columns'] += len(result.get('column_profiles', []))
        summary['total_anomalies'] += len(result.get('anomalies', []))
        summary['total_recommendations'] += len(result.get('recommendations', []))
        
        # Calculate quality score
        validation_results = result.get('validation_results', {})
        if validation_results and 'summary' in validation_results:
            quality_score = validation_results['summary'].get('avg_quality_score', 0.0)
            quality_scores.append(quality_score)
    
    if quality_scores:
        summary['avg_quality_score'] = sum(quality_scores) / len(quality_scores)
    
    return summary


def profile_dataframe(df: pd.DataFrame, 
                     table_name: str = "dataframe",
                     validation_rules: Optional[Dict] = None,
                     output_dir: str = "reports") -> Dict[str, Any]:
    """
    Profile a pandas DataFrame directly
    
    Args:
        df: Input DataFrame
        table_name: Name for the table
        validation_rules: Data validation rules
        output_dir: Output directory for reports
        
    Returns:
        Dictionary with profiling results
    """
    print(f"Profiling DataFrame: {table_name}")
    
    # Initialize components
    column_profiler = ColumnProfiler()
    report_generator = ReportGenerator()
    
    try:
        # Profile columns
        print("  Profiling columns...")
        column_profiles = column_profiler.profile(df)
        print(f"  Profiled {len(column_profiles)} columns")
        
        # Correlation analysis
        print("  Analyzing correlations...")
        correlation_engine = CorrelationEngine(df)
        correlation_results = correlation_engine.generate_report()
        
        # Data validation
        print("  Validating data...")
        validator = DataValidator(df, validation_rules or {})
        validation_results = validator.generate_validation_report()
        
        # Generate reports
        print("  Generating reports...")
        anomalies = column_profiler.find_anomalies()
        recommendations = validator.suggest_improvements()
        
        report_paths = report_generator.generate_comprehensive_report(
            table_name=table_name,
            column_profiles=column_profiles,
            correlations=correlation_results,
            validation_results=validation_results,
            anomalies=anomalies,
            recommendations=recommendations,
            sample_size=len(df),
            output_dir=output_dir
        )
        
        result = {
            'sample_size': len(df),
            'column_profiles': [p.to_dict() for p in column_profiles],
            'correlation_results': correlation_results,
            'validation_results': validation_results,
            'anomalies': anomalies,
            'recommendations': recommendations,
            'report_paths': report_paths,
            'processing_time': datetime.now().isoformat()
        }
        
        print(f"  Completed profiling {table_name}")
        return result
        
    except Exception as e:
        print(f"  Error profiling DataFrame: {e}")
        raise


def load_config(config_file: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise Exception(f"Failed to load configuration file {config_file}: {e}")


def save_results(results: Dict[str, Any], output_file: str) -> None:
    """
    Save results to JSON file
    
    Args:
        results: Results dictionary
        output_file: Output file path
    """
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to: {output_file}")
    except Exception as e:
        print(f"Failed to save results: {e}")


# Example usage functions
def profile_aws_redshift(config: Dict) -> Dict[str, Any]:
    """Profile AWS Redshift tables"""
    return orchestrate({
        'cloud': 'aws',
        'db_type': 'redshift',
        **config
    })


def profile_gcp_bigquery(config: Dict) -> Dict[str, Any]:
    """Profile GCP BigQuery tables"""
    return orchestrate({
        'cloud': 'gcp',
        **config
    })


def profile_azure_sql(config: Dict) -> Dict[str, Any]:
    """Profile Azure SQL Database tables"""
    return orchestrate({
        'cloud': 'azure',
        'db_type': 'sql_database',
        **config
    }) 