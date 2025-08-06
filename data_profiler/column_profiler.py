"""
Column profiling module with YData and Great Expectations integration
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from collections import Counter
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

try:
    from ydata_profiling import ProfileReport
    YDATA_AVAILABLE = True
except ImportError:
    YDATA_AVAILABLE = False

try:
    import great_expectations as ge
    GREAT_EXPECTATIONS_AVAILABLE = True
except ImportError:
    GREAT_EXPECTATIONS_AVAILABLE = False


class ColumnProfile:
    """Represents the profile of a single column"""
    
    def __init__(self, name: str, dtype: str, null_ratio: float, 
                 unique_values: int, value_range: Optional[tuple] = None, 
                 clusters: Optional[Dict] = None, statistics: Optional[Dict] = None):
        self.name = name
        self.dtype = dtype
        self.null_ratio = null_ratio
        self.unique_values = unique_values
        self.value_range = value_range
        self.clusters = clusters
        self.statistics = statistics or {}
    
    def to_dict(self) -> Dict:
        """Convert profile to dictionary"""
        return {
            "name": self.name,
            "dtype": self.dtype,
            "null_ratio": self.null_ratio,
            "unique_values": self.unique_values,
            "value_range": self.value_range,
            "clusters": self.clusters,
            "statistics": self.statistics,
        }


class ColumnProfiler:
    """Comprehensive column profiling with multiple analysis methods"""
    
    def __init__(self):
        self.profiles = []
    
    def profile(self, df: pd.DataFrame) -> List[ColumnProfile]:
        """
        Profile all columns in the DataFrame
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of ColumnProfile objects
        """
        self.profiles.clear()
        
        for col in df.columns:
            series = df[col].dropna()
            null_ratio = 1.0 - len(series) / len(df)
            dtype = str(df[col].dtype)
            unique_values = series.nunique()
            
            # Calculate value range for numeric columns
            value_range = None
            if pd.api.types.is_numeric_dtype(series):
                value_range = (float(series.min()), float(series.max()))
            
            # Find clusters
            clusters = None
            if pd.api.types.is_numeric_dtype(series):
                clusters = self._find_numeric_clusters(series)
            elif unique_values < 100:  # treat as enum
                clusters = self._find_enum_clusters(series)
            
            # Calculate additional statistics
            statistics = self._calculate_statistics(series)
            
            profile = ColumnProfile(
                name=col,
                dtype=dtype,
                null_ratio=null_ratio,
                unique_values=unique_values,
                value_range=value_range,
                clusters=clusters,
                statistics=statistics,
            )
            self.profiles.append(profile)
        
        return self.profiles
    
    def _find_numeric_clusters(self, series: pd.Series) -> Optional[List[float]]:
        """Find clusters in numeric data using K-means"""
        if len(series) < 2:
            return None
        
        try:
            values = series.values.reshape(-1, 1)
            n_clusters = min(5, len(values), len(series.unique()))
            
            if n_clusters < 2:
                return None
            
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            labels = kmeans.fit_predict(values)
            cluster_centers = kmeans.cluster_centers_.flatten().tolist()
            return sorted(cluster_centers)
        except Exception:
            return None
    
    def _find_enum_clusters(self, series: pd.Series) -> Optional[Dict]:
        """Find value distribution for categorical data"""
        try:
            counter = Counter(series)
            return dict(counter.most_common(10))  # Top 10 values
        except Exception:
            return None
    
    def _calculate_statistics(self, series: pd.Series) -> Dict:
        """Calculate comprehensive statistics for a column"""
        stats = {}
        
        if pd.api.types.is_numeric_dtype(series):
            stats.update({
                "mean": float(series.mean()) if len(series) > 0 else None,
                "std": float(series.std()) if len(series) > 0 else None,
                "median": float(series.median()) if len(series) > 0 else None,
                "skewness": float(series.skew()) if len(series) > 0 else None,
                "kurtosis": float(series.kurtosis()) if len(series) > 0 else None,
                "percentiles": {
                    "1%": float(series.quantile(0.01)) if len(series) > 0 else None,
                    "25%": float(series.quantile(0.25)) if len(series) > 0 else None,
                    "50%": float(series.quantile(0.50)) if len(series) > 0 else None,
                    "75%": float(series.quantile(0.75)) if len(series) > 0 else None,
                    "99%": float(series.quantile(0.99)) if len(series) > 0 else None,
                }
            })
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
            # String statistics
            if len(series) > 0:
                str_lengths = series.astype(str).str.len()
                stats.update({
                    "avg_length": float(str_lengths.mean()),
                    "min_length": int(str_lengths.min()),
                    "max_length": int(str_lengths.max()),
                    "top_values": series.value_counts().head(5).to_dict()
                })
        
        # Common statistics for all types
        stats.update({
            "total_count": len(series),
            "null_count": series.isnull().sum(),
            "unique_ratio": series.nunique() / len(series) if len(series) > 0 else 0
        })
        
        return stats
    
    def to_dicts(self) -> List[Dict]:
        """Convert all profiles to dictionaries"""
        return [profile.to_dict() for profile in self.profiles]
    
    def generate_ydata_report(self, df: pd.DataFrame, output_file: str = "ydata_report.html") -> str:
        """
        Generate YData profiling report
        
        Args:
            df: Input DataFrame
            output_file: Output file path
            
        Returns:
            Path to generated report
        """
        if not YDATA_AVAILABLE:
            raise ImportError("YData profiling is not available. Install with: pip install ydata-profiling")
        
        try:
            profile = ProfileReport(df, title="YData Profiling Report", explorative=True)
            profile.to_file(output_file=output_file)
            return output_file
        except Exception as e:
            print(f"Error generating YData report: {e}")
            return None
    
    def validate_with_great_expectations(self, df: pd.DataFrame) -> Dict:
        """
        Validate data using Great Expectations
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with validation results
        """
        if not GREAT_EXPECTATIONS_AVAILABLE:
            raise ImportError("Great Expectations is not available. Install with: pip install great-expectations")
        
        try:
            ge_df = ge.dataset.PandasDataset(df)
            results = {}
            
            for column in df.columns:
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
                    column_results["in_type"] = ge_df.expect_column_values_to_be_of_type(column, str(df[column].dtype)).success
                except:
                    column_results["in_type"] = None
                
                # Numeric-specific expectations
                if pd.api.types.is_numeric_dtype(df[column]):
                    try:
                        column_results["between"] = ge_df.expect_column_values_to_be_between(
                            column, 
                            min_value=df[column].min(), 
                            max_value=df[column].max()
                        ).success
                    except:
                        column_results["between"] = None
                
                results[column] = column_results
            
            return results
        except Exception as e:
            print(f"Error running Great Expectations validation: {e}")
            return {}
    
    def get_summary(self) -> Dict:
        """Get summary statistics across all columns"""
        if not self.profiles:
            return {}
        
        summary = {
            "total_columns": len(self.profiles),
            "numeric_columns": len([p for p in self.profiles if pd.api.types.is_numeric_dtype(p.dtype)]),
            "categorical_columns": len([p for p in self.profiles if not pd.api.types.is_numeric_dtype(p.dtype)]),
            "columns_with_nulls": len([p for p in self.profiles if p.null_ratio > 0]),
            "avg_null_ratio": np.mean([p.null_ratio for p in self.profiles]),
            "columns_with_clusters": len([p for p in self.profiles if p.clusters is not None]),
        }
        
        return summary
    
    def find_anomalies(self) -> List[Dict]:
        """Find potential data quality anomalies"""
        anomalies = []
        
        for profile in self.profiles:
            # High null ratio
            if profile.null_ratio > 0.5:
                anomalies.append({
                    "column": profile.name,
                    "type": "high_null_ratio",
                    "value": profile.null_ratio,
                    "description": f"Column has {profile.null_ratio:.1%} null values"
                })
            
            # Low unique ratio (potential duplicate data)
            if profile.statistics.get("unique_ratio", 1) < 0.1:
                anomalies.append({
                    "column": profile.name,
                    "type": "low_uniqueness",
                    "value": profile.statistics.get("unique_ratio", 0),
                    "description": f"Column has low uniqueness ({profile.statistics.get('unique_ratio', 0):.1%})"
                })
            
            # Extreme values in numeric columns
            if profile.statistics.get("skewness") and abs(profile.statistics["skewness"]) > 3:
                anomalies.append({
                    "column": profile.name,
                    "type": "high_skewness",
                    "value": profile.statistics["skewness"],
                    "description": f"Column has high skewness ({profile.statistics['skewness']:.2f})"
                })
        
        return anomalies 