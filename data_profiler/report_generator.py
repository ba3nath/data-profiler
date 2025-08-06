"""
Report generator module with Jinja2 templates
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader, Template
import pandas as pd


class ReportGenerator:
    """Generate comprehensive HTML and markdown reports"""
    
    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize report generator
        
        Args:
            template_dir: Directory containing Jinja2 templates
        """
        if template_dir and os.path.exists(template_dir):
            self.env = Environment(loader=FileSystemLoader(template_dir))
        else:
            # Use default template
            self.env = Environment()
            self._setup_default_templates()
    
    def _setup_default_templates(self):
        """Setup default Jinja2 templates"""
        self.html_template = Template("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Profiling Report - {{ table_name }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; border-bottom: 2px solid #007bff; padding-bottom: 20px; margin-bottom: 30px; }
        .section { margin-bottom: 30px; }
        .section h2 { color: #007bff; border-bottom: 1px solid #ddd; padding-bottom: 10px; }
        .metric-card { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 5px; padding: 15px; margin: 10px 0; }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; }
        .correlation-table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        .correlation-table th, .correlation-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        .correlation-table th { background-color: #007bff; color: white; }
        .correlation-table tr:nth-child(even) { background-color: #f2f2f2; }
        .alert { padding: 10px; border-radius: 5px; margin: 10px 0; }
        .alert-warning { background-color: #fff3cd; border: 1px solid #ffeaa7; color: #856404; }
        .alert-danger { background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
        .alert-success { background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
        .summary-stats { display: flex; justify-content: space-around; flex-wrap: wrap; margin: 20px 0; }
        .stat-box { text-align: center; padding: 15px; background: #e9ecef; border-radius: 5px; margin: 5px; min-width: 120px; }
        .stat-number { font-size: 24px; font-weight: bold; color: #007bff; }
        .stat-label { font-size: 14px; color: #6c757d; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Data Profiling Report</h1>
            <p><strong>Table:</strong> {{ table_name }}</p>
            <p><strong>Generated:</strong> {{ generation_time }}</p>
            <p><strong>Sample Size:</strong> {{ sample_size }} rows</p>
        </div>

        <div class="section">
            <h2>Summary Statistics</h2>
            <div class="summary-stats">
                <div class="stat-box">
                    <div class="stat-number">{{ summary.total_columns }}</div>
                    <div class="stat-label">Total Columns</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{{ summary.numeric_columns }}</div>
                    <div class="stat-label">Numeric Columns</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{{ summary.categorical_columns }}</div>
                    <div class="stat-label">Categorical Columns</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{{ "%.1f"|format(summary.avg_null_ratio * 100) }}%</div>
                    <div class="stat-label">Avg Null Ratio</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Column Profiles</h2>
            {% for profile in column_profiles %}
            <div class="metric-card">
                <h3>{{ profile.name }}</h3>
                <p><strong>Type:</strong> {{ profile.dtype }}</p>
                <p><strong>Null Ratio:</strong> {{ "%.2f"|format(profile.null_ratio * 100) }}%</p>
                <p><strong>Unique Values:</strong> {{ profile.unique_values }}</p>
                {% if profile.value_range %}
                <p><strong>Value Range:</strong> {{ profile.value_range[0] }} to {{ profile.value_range[1] }}</p>
                {% endif %}
                {% if profile.statistics %}
                <div class="metric-grid">
                    {% for stat_name, stat_value in profile.statistics.items() %}
                    {% if stat_value is not none %}
                    <div><strong>{{ stat_name }}:</strong> {{ stat_value }}</div>
                    {% endif %}
                    {% endfor %}
                </div>
                {% endif %}
            </div>
            {% endfor %}
        </div>

        {% if correlations %}
        <div class="section">
            <h2>Correlation Analysis</h2>
            {% if correlations.top_correlations %}
            <h3>Top Correlations</h3>
            <table class="correlation-table">
                <thead>
                    <tr>
                        <th>Column 1</th>
                        <th>Column 2</th>
                        <th>Correlation</th>
                    </tr>
                </thead>
                <tbody>
                    {% for col1, col2, corr in correlations.top_correlations[:10] %}
                    <tr>
                        <td>{{ col1 }}</td>
                        <td>{{ col2 }}</td>
                        <td>{{ "%.3f"|format(corr) }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% endif %}
        </div>
        {% endif %}

        {% if validation_results %}
        <div class="section">
            <h2>Data Validation Results</h2>
            {% if validation_results.summary.total_violations > 0 %}
            <div class="alert alert-warning">
                <strong>Warning:</strong> {{ validation_results.summary.total_violations }} validation violations found.
            </div>
            {% else %}
            <div class="alert alert-success">
                <strong>Success:</strong> All validation checks passed.
            </div>
            {% endif %}
            
            <div class="metric-grid">
                <div class="metric-card">
                    <h4>Quality Metrics</h4>
                    <p><strong>Average Quality Score:</strong> {{ "%.2f"|format(validation_results.summary.avg_quality_score * 100) }}%</p>
                    <p><strong>Columns with Violations:</strong> {{ validation_results.summary.columns_with_violations }}</p>
                </div>
            </div>
        </div>
        {% endif %}

        {% if anomalies %}
        <div class="section">
            <h2>Data Quality Anomalies</h2>
            {% for anomaly in anomalies %}
            <div class="alert alert-danger">
                <strong>{{ anomaly.type }}:</strong> {{ anomaly.description }}
            </div>
            {% endfor %}
        </div>
        {% endif %}

        <div class="section">
            <h2>Recommendations</h2>
            {% if recommendations %}
            <ul>
                {% for recommendation in recommendations %}
                <li>{{ recommendation }}</li>
                {% endfor %}
            </ul>
            {% else %}
            <p>No specific recommendations at this time.</p>
            {% endif %}
        </div>
    </div>
</body>
</html>
        """)
        
        self.markdown_template = Template("""
# Data Profiling Report

**Table:** {{ table_name }}  
**Generated:** {{ generation_time }}  
**Sample Size:** {{ sample_size }} rows

## Summary Statistics

- **Total Columns:** {{ summary.total_columns }}
- **Numeric Columns:** {{ summary.numeric_columns }}
- **Categorical Columns:** {{ summary.categorical_columns }}
- **Average Null Ratio:** {{ "%.1f"|format(summary.avg_null_ratio * 100) }}%
- **Columns with Clusters:** {{ summary.columns_with_clusters }}

## Column Profiles

{% for profile in column_profiles %}
### {{ profile.name }}

- **Type:** {{ profile.dtype }}
- **Null Ratio:** {{ "%.2f"|format(profile.null_ratio * 100) }}%
- **Unique Values:** {{ profile.unique_values }}
{% if profile.value_range %}
- **Value Range:** {{ profile.value_range[0] }} to {{ profile.value_range[1] }}
{% endif %}

{% if profile.statistics %}
**Statistics:**
{% for stat_name, stat_value in profile.statistics.items() %}
{% if stat_value is not none %}
- {{ stat_name }}: {{ stat_value }}
{% endif %}
{% endfor %}
{% endif %}

{% endfor %}

{% if correlations %}
## Correlation Analysis

{% if correlations.top_correlations %}
### Top Correlations

| Column 1 | Column 2 | Correlation |
|----------|----------|-------------|
{% for col1, col2, corr in correlations.top_correlations[:10] %}
| {{ col1 }} | {{ col2 }} | {{ "%.3f"|format(corr) }} |
{% endfor %}
{% endif %}
{% endif %}

{% if validation_results %}
## Data Validation Results

{% if validation_results.summary.total_violations > 0 %}
⚠️ **Warning:** {{ validation_results.summary.total_violations }} validation violations found.
{% else %}
✅ **Success:** All validation checks passed.
{% endif %}

- **Average Quality Score:** {{ "%.2f"|format(validation_results.summary.avg_quality_score * 100) }}%
- **Columns with Violations:** {{ validation_results.summary.columns_with_violations }}
- **Outlier Columns:** {{ validation_results.summary.outlier_columns }}
{% endif %}

{% if anomalies %}
## Data Quality Anomalies

{% for anomaly in anomalies %}
- **{{ anomaly.type }}:** {{ anomaly.description }}
{% endfor %}
{% endif %}

## Recommendations

{% if recommendations %}
{% for recommendation in recommendations %}
- {{ recommendation }}
{% endfor %}
{% else %}
No specific recommendations at this time.
{% endif %}
        """)
    
    def generate_html_report(self, context: Dict[str, Any], output_file: Optional[str] = None) -> str:
        """
        Generate HTML report
        
        Args:
            context: Dictionary with report data
            output_file: Output file path
            
        Returns:
            Generated HTML content or file path
        """
        # Add generation time if not present
        if 'generation_time' not in context:
            context['generation_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Render template
        if hasattr(self, 'html_template'):
            html_content = self.html_template.render(**context)
        else:
            template = self.env.get_template('report.html')
            html_content = template.render(**context)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            return output_file
        else:
            return html_content
    
    def generate_markdown_report(self, context: Dict[str, Any], output_file: Optional[str] = None) -> str:
        """
        Generate markdown report
        
        Args:
            context: Dictionary with report data
            output_file: Output file path
            
        Returns:
            Generated markdown content or file path
        """
        # Add generation time if not present
        if 'generation_time' not in context:
            context['generation_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Render template
        if hasattr(self, 'markdown_template'):
            md_content = self.markdown_template.render(**context)
        else:
            template = self.env.get_template('report.md')
            md_content = template.render(**context)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(md_content)
            return output_file
        else:
            return md_content
    
    def generate_json_report(self, context: Dict[str, Any], output_file: Optional[str] = None) -> str:
        """
        Generate JSON report
        
        Args:
            context: Dictionary with report data
            output_file: Output file path
            
        Returns:
            Generated JSON content or file path
        """
        # Add generation time if not present
        if 'generation_time' not in context:
            context['generation_time'] = datetime.now().isoformat()
        
        # Convert any non-serializable objects
        json_context = self._prepare_json_context(context)
        
        json_content = json.dumps(json_context, indent=2, default=str)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(json_content)
            return output_file
        else:
            return json_content
    
    def _prepare_json_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare context for JSON serialization"""
        json_context = {}
        
        for key, value in context.items():
            if isinstance(value, pd.DataFrame):
                json_context[key] = value.to_dict('records')
            elif isinstance(value, pd.Series):
                json_context[key] = value.to_dict()
            elif isinstance(value, dict):
                json_context[key] = self._prepare_json_context(value)
            elif isinstance(value, list):
                json_context[key] = [
                    self._prepare_json_context(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                json_context[key] = value
        
        return json_context
    
    def generate_comprehensive_report(self, 
                                    table_name: str,
                                    column_profiles: list,
                                    correlations: Optional[Dict] = None,
                                    validation_results: Optional[Dict] = None,
                                    anomalies: Optional[list] = None,
                                    recommendations: Optional[list] = None,
                                    sample_size: int = 0,
                                    output_dir: str = "reports") -> Dict[str, str]:
        """
        Generate comprehensive report in multiple formats
        
        Args:
            table_name: Name of the table being profiled
            column_profiles: List of column profile objects
            correlations: Correlation analysis results
            validation_results: Data validation results
            anomalies: List of data quality anomalies
            recommendations: List of recommendations
            sample_size: Number of rows sampled
            output_dir: Output directory for reports
            
        Returns:
            Dictionary with paths to generated reports
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare context
        context = {
            'table_name': table_name,
            'column_profiles': [profile.to_dict() if hasattr(profile, 'to_dict') else profile for profile in column_profiles],
            'correlations': correlations or {},
            'validation_results': validation_results or {},
            'anomalies': anomalies or [],
            'recommendations': recommendations or [],
            'sample_size': sample_size,
            'summary': self._generate_summary(column_profiles, validation_results)
        }
        
        # Generate reports
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{table_name}_{timestamp}"
        
        reports = {}
        
        # HTML report
        html_file = os.path.join(output_dir, f"{base_filename}.html")
        reports['html'] = self.generate_html_report(context, html_file)
        
        # Markdown report
        md_file = os.path.join(output_dir, f"{base_filename}.md")
        reports['markdown'] = self.generate_markdown_report(context, md_file)
        
        # JSON report
        json_file = os.path.join(output_dir, f"{base_filename}.json")
        reports['json'] = self.generate_json_report(context, json_file)
        
        return reports
    
    def _generate_summary(self, column_profiles: list, validation_results: Optional[Dict]) -> Dict:
        """Generate summary statistics"""
        summary = {
            'total_columns': len(column_profiles),
            'numeric_columns': 0,
            'categorical_columns': 0,
            'columns_with_nulls': 0,
            'avg_null_ratio': 0.0,
            'columns_with_clusters': 0
        }
        
        if column_profiles:
            null_ratios = []
            for profile in column_profiles:
                profile_dict = profile.to_dict() if hasattr(profile, 'to_dict') else profile
                
                if pd.api.types.is_numeric_dtype(profile_dict.get('dtype', '')):
                    summary['numeric_columns'] += 1
                else:
                    summary['categorical_columns'] += 1
                
                null_ratio = profile_dict.get('null_ratio', 0)
                null_ratios.append(null_ratio)
                
                if null_ratio > 0:
                    summary['columns_with_nulls'] += 1
                
                if profile_dict.get('clusters'):
                    summary['columns_with_clusters'] += 1
            
            summary['avg_null_ratio'] = sum(null_ratios) / len(null_ratios) if null_ratios else 0
        
        return summary 