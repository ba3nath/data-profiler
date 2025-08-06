from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="data-profiler",
    version="1.0.0",
    author="Data Profiler Team",
    author_email="team@dataprofiler.com",
    description="A comprehensive data profiling and validation system for cloud databases",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/data-profiler",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Data Scientists",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Database",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "full": [
            "openmetadata-ingestion>=1.0.0",
            "great-expectations>=0.17.0",
            "ydata-profiling>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "data-profiler=data_profiler.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "data_profiler": ["templates/*.html", "templates/*.md"],
    },
    keywords="data profiling, data quality, data validation, database, analytics",
    project_urls={
        "Bug Reports": "https://github.com/your-org/data-profiler/issues",
        "Source": "https://github.com/your-org/data-profiler",
        "Documentation": "https://github.com/your-org/data-profiler/blob/main/README.md",
    },
) 