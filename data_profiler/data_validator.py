"""
Data validation module with Great Expectations integration
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, Any, List, Optional
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

try:
    import great_expectations as ge
    GREAT_EXPECTATIONS_AVAILABLE = True
except ImportError:
    GREAT_EXPECTATIONS_AVAILABLE = False


class DataValidator:
    """Comprehensive data validation with custom rules and Great Expectations"""
    
    def __init__(self, df: pd.DataFrame, rules: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize data validator
        
        Args:
            df: Input DataFrame
            rules: Dictionary of validation rules per column
        """
        self.df = df.copy()
        self.rules = rules or {}
        self.violations = {}
        self.validation_results = {}
    
    def validate_rules(self) -> Dict[str, Dict]:
        """
        Validate data against custom rules
        
        Returns:
            Dictionary with validation results
        """
        self.violations.clear()
        
        for col, rule in self.rules.items():
            if col not in self.df.columns:
                continue
            
            column_violations = []
            
            # Range validation
            if 'min' in rule:
                min_violations = self.df[col] < rule['min']
                if min_violations.any():
                    column_violations.append({
                        'type': 'below_min',
                        'count': min_violations.sum(),
                        'threshold': rule['min']
                    })
            
            if 'max' in rule:
                max_violations = self.df[col] > rule['max']
                if max_violations.any():
                    column_violations.append({
                        'type': 'above_max',
                        'count': max_violations.sum(),
                        'threshold': rule['max']
                    })
            
            # Pattern validation
            if 'pattern' in rule:
                pattern_violations = ~self.df[col].astype(str).str.match(rule['pattern'], na=False)
                if pattern_violations.any():
                    column_violations.append({
                        'type': 'pattern_mismatch',
                        'count': pattern_violations.sum(),
                        'pattern': rule['pattern']
                    })
            
            # Unique validation
            if rule.get('unique', False):
                duplicate_count = self.df[col].duplicated().sum()
                if duplicate_count > 0:
                    column_violations.append({
                        'type': 'duplicates',
                        'count': duplicate_count
                    })
            
            # Null validation
            if rule.get('not_null', False):
                null_count = self.df[col].isnull().sum()
                if null_count > 0:
                    column_violations.append({
                        'type': 'null_values',
                        'count': null_count
                    })
            
            # Enum validation
            if 'allowed_values' in rule:
                invalid_values = ~self.df[col].isin(rule['allowed_values'])
                if invalid_values.any():
                    column_violations.append({
                        'type': 'invalid_enum',
                        'count': invalid_values.sum(),
                        'allowed_values': rule['allowed_values']
                    })
            
            if column_violations:
                self.violations[col] = column_violations
        
        return self.violations
    
    def detect_clusters(self, method: str = "kmeans", n_clusters: int = 5) -> pd.Series:
        """
        Detect clusters in the data
        
        Args:
            method: Clustering method ('kmeans', 'dbscan')
            n_clusters: Number of clusters for K-means
            
        Returns:
            Series with cluster labels
        """
        df_encoded = self._encode_data()
        
        if method == "kmeans":
            model = KMeans(n_clusters=min(n_clusters, len(df_encoded)), random_state=42)
        elif method == "dbscan":
            model = DBSCAN(eps=0.5, min_samples=5)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
        
        try:
            labels = model.fit_predict(df_encoded)
            return pd.Series(labels, index=self.df.index, name="cluster")
        except Exception as e:
            print(f"Clustering failed: {e}")
            return pd.Series([0] * len(self.df), index=self.df.index, name="cluster")
    
    def _encode_data(self) -> pd.DataFrame:
        """Encode categorical data for clustering"""
        df = self.df.copy()
        
        for col in df.select_dtypes(include=["object", "category"]).columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        
        # Fill NaN values with 0 for clustering
        df = df.fillna(0)
        
        # Standardize the data
        scaler = StandardScaler()
        return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    def detect_outliers(self, method: str = "iqr", threshold: float = 1.5) -> Dict[str, List[int]]:
        """
        Detect outliers in numeric columns
        
        Args:
            method: Outlier detection method ('iqr', 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Dictionary with outlier indices per column
        """
        outliers = {}
        
        for col in self.df.select_dtypes(include=[np.number]).columns:
            series = self.df[col].dropna()
            
            if method == "iqr":
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            
            elif method == "zscore":
                z_scores = np.abs((series - series.mean()) / series.std())
                outlier_mask = z_scores > threshold
            
            else:
                continue
            
            outlier_indices = self.df[outlier_mask].index.tolist()
            if outlier_indices:
                outliers[col] = outlier_indices
        
        return outliers
    
    def validate_with_great_expectations(self) -> Dict:
        """
        Validate data using Great Expectations
        
        Returns:
            Dictionary with Great Expectations validation results
        """
        if not GREAT_EXPECTATIONS_AVAILABLE:
            raise ImportError("Great Expectations is not available. Install with: pip install great-expectations")
        
        try:
            ge_df = ge.dataset.PandasDataset(self.df)
            results = {}
            
            for column in self.df.columns:
                column_results = {}
                
                # Basic expectations
                try:
                    column_results["not_null"] = ge_df.expect_column_values_to_not_be_null(column).success
                except:
                    column_results["not_null"] = None
                
                try:
                    column_results["unique"] = ge_df.expect_column_values_to_be_unique(column).success
                except:
                    column_results["unique"] = None
                
                try:
                    column_results["in_type"] = ge_df.expect_column_values_to_be_of_type(column, str(self.df[column].dtype)).success
                except:
                    column_results["in_type"] = None
                
                # Numeric-specific expectations
                if pd.api.types.is_numeric_dtype(self.df[column]):
                    try:
                        column_results["between"] = ge_df.expect_column_values_to_be_between(
                            column, 
                            min_value=self.df[column].min(), 
                            max_value=self.df[column].max()
                        ).success
                    except:
                        column_results["between"] = None
                    
                    try:
                        column_results["not_null"] = ge_df.expect_column_values_to_not_be_null(column).success
                    except:
                        column_results["not_null"] = None
                
                # String-specific expectations
                elif pd.api.types.is_object_dtype(self.df[column]):
                    try:
                        column_results["string_length_between"] = ge_df.expect_column_value_lengths_to_be_between(
                            column, min_value=0, max_value=1000
                        ).success
                    except:
                        column_results["string_length_between"] = None
                
                results[column] = column_results
            
            self.validation_results['great_expectations'] = results
            return results
        
        except Exception as e:
            print(f"Error running Great Expectations validation: {e}")
            return {}
    
    def check_data_quality_metrics(self) -> Dict:
        """
        Calculate comprehensive data quality metrics
        
        Returns:
            Dictionary with quality metrics
        """
        quality_metrics = {}
        
        for col in self.df.columns:
            series = self.df[col]
            
            metrics = {
                'completeness': 1 - series.isnull().sum() / len(series),
                'uniqueness': series.nunique() / len(series),
                'consistency': self._calculate_consistency(series),
                'validity': self._calculate_validity(series)
            }
            
            quality_metrics[col] = metrics
        
        return quality_metrics
    
    def _calculate_consistency(self, series: pd.Series) -> float:
        """Calculate data consistency score"""
        if pd.api.types.is_numeric_dtype(series):
            # For numeric data, check if values are within reasonable bounds
            if series.std() == 0:
                return 1.0  # All values are the same
            z_scores = np.abs((series - series.mean()) / series.std())
            return 1 - (z_scores > 3).sum() / len(series)
        else:
            # For categorical data, check if there are too many unique values
            unique_ratio = series.nunique() / len(series)
            return 1 - unique_ratio if unique_ratio > 0.5 else unique_ratio
    
    def _calculate_validity(self, series: pd.Series) -> float:
        """Calculate data validity score"""
        # This is a simplified validity check
        # In practice, you would implement domain-specific validation rules
        if pd.api.types.is_numeric_dtype(series):
            # Check for infinite values
            infinite_count = np.isinf(series).sum()
            return 1 - infinite_count / len(series)
        else:
            # Check for empty strings
            empty_count = (series.astype(str).str.strip() == '').sum()
            return 1 - empty_count / len(series)
    
    def generate_validation_report(self) -> Dict:
        """
        Generate comprehensive validation report
        
        Returns:
            Dictionary with all validation results
        """
        # Run all validations
        custom_violations = self.validate_rules()
        ge_results = self.validate_with_great_expectations()
        quality_metrics = self.check_data_quality_metrics()
        outliers = self.detect_outliers()
        clusters = self.detect_clusters()
        
        # Calculate summary statistics
        total_violations = sum(len(violations) for violations in custom_violations.values())
        avg_quality_score = np.mean([
            np.mean(list(metrics.values())) 
            for metrics in quality_metrics.values()
        ])
        
        report = {
            "custom_rule_violations": custom_violations,
            "great_expectations_results": ge_results,
            "quality_metrics": quality_metrics,
            "outliers": outliers,
            "cluster_distribution": clusters.value_counts().to_dict(),
            "summary": {
                "total_violations": total_violations,
                "avg_quality_score": avg_quality_score,
                "columns_with_violations": len(custom_violations),
                "outlier_columns": len(outliers),
                "total_rows": len(self.df),
                "total_columns": len(self.df.columns)
            }
        }
        
        return report
    
    def suggest_improvements(self) -> List[str]:
        """
        Suggest data quality improvements based on validation results
        
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        # Analyze violations
        for col, violations in self.violations.items():
            for violation in violations:
                if violation['type'] == 'below_min':
                    suggestions.append(f"Column '{col}' has {violation['count']} values below minimum threshold")
                elif violation['type'] == 'above_max':
                    suggestions.append(f"Column '{col}' has {violation['count']} values above maximum threshold")
                elif violation['type'] == 'pattern_mismatch':
                    suggestions.append(f"Column '{col}' has {violation['count']} values that don't match expected pattern")
                elif violation['type'] == 'duplicates':
                    suggestions.append(f"Column '{col}' has {violation['count']} duplicate values")
                elif violation['type'] == 'null_values':
                    suggestions.append(f"Column '{col}' has {violation['count']} null values")
        
        # Analyze quality metrics
        quality_metrics = self.check_data_quality_metrics()
        for col, metrics in quality_metrics.items():
            if metrics['completeness'] < 0.9:
                suggestions.append(f"Column '{col}' has low completeness ({metrics['completeness']:.1%})")
            if metrics['uniqueness'] < 0.1:
                suggestions.append(f"Column '{col}' has low uniqueness ({metrics['uniqueness']:.1%})")
            if metrics['consistency'] < 0.8:
                suggestions.append(f"Column '{col}' has low consistency ({metrics['consistency']:.1%})")
        
        return suggestions[:10]  # Return top 10 suggestions 