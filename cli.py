#!/usr/bin/env python3
"""
Command-line interface for Data Profiler
"""

import argparse
import json
import sys
import os
from pathlib import Path

from data_profiler import orchestrate, profile_dataframe, load_config, save_results
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Data Profiler - Comprehensive data profiling and validation tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile a CSV file
  python cli.py profile-file data.csv --output-dir reports

  # Profile database tables using config file
  python cli.py profile-db config.json

  # Profile DataFrame with custom validation rules
  python cli.py profile-file data.csv --validation-rules rules.json

  # Run demo with sample data
  python cli.py demo
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Profile file command
    file_parser = subparsers.add_parser('profile-file', help='Profile a data file (CSV, Excel, etc.)')
    file_parser.add_argument('file', help='Path to the data file')
    file_parser.add_argument('--output-dir', default='reports', help='Output directory for reports')
    file_parser.add_argument('--validation-rules', help='Path to validation rules JSON file')
    file_parser.add_argument('--table-name', default=None, help='Name for the table')
    file_parser.add_argument('--sample-size', type=int, default=None, help='Number of rows to sample')
    
    # Profile database command
    db_parser = subparsers.add_parser('profile-db', help='Profile database tables')
    db_parser.add_argument('config', help='Path to configuration JSON file')
    db_parser.add_argument('--output-file', help='Path to save results JSON file')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demo with sample data')
    demo_parser.add_argument('--output-dir', default='demo_reports', help='Output directory for demo reports')
    
    # List supported formats
    formats_parser = subparsers.add_parser('formats', help='List supported file formats')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'profile-file':
            profile_file_command(args)
        elif args.command == 'profile-db':
            profile_db_command(args)
        elif args.command == 'demo':
            demo_command(args)
        elif args.command == 'formats':
            formats_command()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def profile_file_command(args):
    """Handle profile-file command"""
    file_path = Path(args.file)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f"Loading data from: {file_path}")
    
    # Load data based on file extension
    if file_path.suffix.lower() == '.csv':
        df = pd.read_csv(file_path)
    elif file_path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    elif file_path.suffix.lower() == '.json':
        df = pd.read_json(file_path)
    elif file_path.suffix.lower() == '.parquet':
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    # Load validation rules if provided
    validation_rules = {}
    if args.validation_rules:
        validation_rules = load_config(args.validation_rules)
    
    # Determine table name
    table_name = args.table_name or file_path.stem
    
    # Sample data if requested
    if args.sample_size and len(df) > args.sample_size:
        df = df.sample(n=args.sample_size, random_state=42)
        print(f"Sampled {len(df)} rows")
    
    # Profile the data
    print("Starting profiling...")
    results = profile_dataframe(
        df=df,
        table_name=table_name,
        validation_rules=validation_rules,
        output_dir=args.output_dir
    )
    
    print(f"\nProfiling completed!")
    print(f"Reports saved to: {args.output_dir}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"- Columns profiled: {len(results['column_profiles'])}")
    print(f"- Anomalies found: {len(results['anomalies'])}")
    print(f"- Recommendations: {len(results['recommendations'])}")
    
    # Save results to JSON
    results_file = Path(args.output_dir) / f"{table_name}_results.json"
    save_results(results, str(results_file))


def profile_db_command(args):
    """Handle profile-db command"""
    config_path = Path(args.config)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    print(f"Loading configuration from: {config_path}")
    
    # Load configuration
    config = load_config(str(config_path))
    
    # Run profiling
    print("Starting database profiling...")
    results = orchestrate(config)
    
    print(f"\nProfiling completed!")
    print(f"Processed {results['summary']['total_tables']} tables")
    print(f"Successful: {results['summary']['successful_tables']}")
    print(f"Failed: {results['summary']['failed_tables']}")
    
    # Save results if requested
    if args.output_file:
        save_results(results, args.output_file)
    else:
        # Save to default location
        output_dir = config.get('output_dir', 'reports')
        results_file = Path(output_dir) / "profiling_results.json"
        save_results(results, str(results_file))


def demo_command(args):
    """Handle demo command"""
    print("Running Data Profiler Demo...")
    
    # Import demo function
    from examples.demo import main as run_demo
    
    # Override output directory
    os.environ['DEMO_OUTPUT_DIR'] = args.output_dir
    
    # Run demo
    run_demo()
    
    print(f"\nDemo completed! Check the '{args.output_dir}' directory for reports.")


def formats_command():
    """List supported file formats"""
    print("Supported file formats:")
    print("- CSV (.csv)")
    print("- Excel (.xlsx, .xls)")
    print("- JSON (.json)")
    print("- Parquet (.parquet)")
    print("\nDatabase connections:")
    print("- AWS Redshift")
    print("- AWS RDS (PostgreSQL, MySQL)")
    print("- Google BigQuery")
    print("- Azure SQL Database")
    print("- Azure Synapse")


if __name__ == "__main__":
    main() 