#!/usr/bin/env python3
"""
Test runner script for Data Profiler
"""

import sys
import subprocess
import argparse
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run Data Profiler tests")
    parser.add_argument(
        "--unit", 
        action="store_true", 
        help="Run unit tests only"
    )
    parser.add_argument(
        "--integration", 
        action="store_true", 
        help="Run integration tests only"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true", 
        help="Generate coverage report"
    )
    parser.add_argument(
        "--fast", 
        action="store_true", 
        help="Skip slow tests"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Verbose output"
    )
    parser.add_argument(
        "--parallel", 
        action="store_true", 
        help="Run tests in parallel"
    )
    
    args = parser.parse_args()
    
    # Base pytest command
    pytest_cmd = ["python", "-m", "pytest"]
    
    # Add options based on arguments
    if args.unit:
        pytest_cmd.extend(["-m", "unit"])
    elif args.integration:
        pytest_cmd.extend(["-m", "integration"])
    
    if args.fast:
        pytest_cmd.extend(["-m", "not slow"])
    
    if args.verbose:
        pytest_cmd.append("-v")
    
    if args.parallel:
        pytest_cmd.extend(["-n", "auto"])
    
    if args.coverage:
        pytest_cmd.extend([
            "--cov=data_profiler",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-report=xml"
        ])
    
    # Run tests
    success = run_command(pytest_cmd, "Test Suite")
    
    if args.coverage and success:
        print(f"\n📊 Coverage report generated:")
        print(f"   HTML: htmlcov/index.html")
        print(f"   XML: coverage.xml")
    
    # Run linting if tests pass
    if success:
        print(f"\n🔍 Running code quality checks...")
        
        # Check if black is available
        try:
            run_command(["black", "--check", "."], "Code Formatting (Black)")
        except FileNotFoundError:
            print("⚠️  Black not found, skipping formatting check")
        
        # Check if flake8 is available
        try:
            run_command(["flake8", "."], "Code Linting (Flake8)")
        except FileNotFoundError:
            print("⚠️  Flake8 not found, skipping linting check")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 