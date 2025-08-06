"""
Correlation analysis engine for bivariate and multivariate relationships
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class CorrelationEngine:
    """Advanced correlation analysis engine"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize correlation engine
        
        Args:
            df: Input DataFrame
        """
        self.df = df.copy()
        self.encoded_df = self._encode_categoricals()
        self.numeric_df = self.encoded_df.select_dtypes(include=[np.number]).dropna()
    
    def _encode_categoricals(self) -> pd.DataFrame:
        """Encode categorical variables for correlation analysis"""
        df = self.df.copy()
        
        for col in df.select_dtypes(include=["object", "category"]).columns:
            try:
                # Try to convert to numeric first
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                # If conversion fails, use label encoding
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        
        return df
    
    def compute_bivariate_correlations(self, method: str = 'spearman') -> Dict[str, Dict[str, float]]:
        """
        Compute bivariate correlations between all pairs of columns
        
        Args:
            method: Correlation method ('spearman', 'pearson')
            
        Returns:
            Dictionary of correlation matrices
        """
        correlations = {}
        
        for col1 in self.encoded_df.columns:
            for col2 in self.encoded_df.columns:
                if col1 >= col2:
                    continue
                
                try:
                    if method == 'spearman':
                        corr, _ = spearmanr(self.encoded_df[col1], self.encoded_df[col2])
                    else:
                        corr, _ = pearsonr(self.encoded_df[col1], self.encoded_df[col2])
                    
                    if not np.isnan(corr):
                        correlations.setdefault(col1, {})[col2] = corr
                except:
                    continue
        
        return correlations
    
    def compute_correlation_matrix(self, method: str = 'spearman') -> pd.DataFrame:
        """
        Compute full correlation matrix
        
        Args:
            method: Correlation method
            
        Returns:
            Correlation matrix as DataFrame
        """
        if method == 'spearman':
            return self.encoded_df.corr(method='spearman')
        else:
            return self.encoded_df.corr(method='pearson')
    
    def top_correlated_pairs(self, threshold: float = 0.7, 
                           method: str = 'spearman') -> List[Tuple[str, str, float]]:
        """
        Find top correlated column pairs
        
        Args:
            threshold: Minimum correlation threshold
            method: Correlation method
            
        Returns:
            List of (col1, col2, correlation) tuples
        """
        corr_matrix = self.compute_correlation_matrix(method)
        top_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                
                if abs(corr_value) >= threshold and not np.isnan(corr_value):
                    top_pairs.append((col1, col2, corr_value))
        
        return sorted(top_pairs, key=lambda x: -abs(x[2]))
    
    def compute_multivariate_components(self, n_components: int = 2) -> pd.DataFrame:
        """
        Perform Principal Component Analysis
        
        Args:
            n_components: Number of principal components
            
        Returns:
            DataFrame with principal components
        """
        if len(self.numeric_df.columns) < n_components:
            n_components = len(self.numeric_df.columns)
        
        if n_components < 1:
            return pd.DataFrame()
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.numeric_df)
        
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(scaled_data)
        
        # Create DataFrame with component names
        component_names = [f"PC{i+1}" for i in range(n_components)]
        pc_df = pd.DataFrame(principal_components, columns=component_names)
        
        # Add explained variance information
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        self.pca_info = {
            'explained_variance': explained_variance.tolist(),
            'cumulative_variance': cumulative_variance.tolist(),
            'feature_importance': dict(zip(self.numeric_df.columns, pca.components_[0]))
        }
        
        return pc_df
    
    def find_feature_clusters(self, n_clusters: int = 3) -> Dict:
        """
        Cluster features based on their correlation patterns
        
        Args:
            n_clusters: Number of clusters
            
        Returns:
            Dictionary with cluster assignments
        """
        if len(self.numeric_df.columns) < n_clusters:
            n_clusters = len(self.numeric_df.columns)
        
        # Use correlation matrix as feature similarity
        corr_matrix = self.compute_correlation_matrix()
        
        # Convert correlation to distance (1 - abs(correlation))
        distance_matrix = 1 - np.abs(corr_matrix.values)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(distance_matrix)
        
        # Group features by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            feature_name = corr_matrix.columns[i]
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(feature_name)
        
        return clusters
    
    def detect_collinearity(self, threshold: float = 0.9) -> List[Tuple[str, str, float]]:
        """
        Detect highly collinear features
        
        Args:
            threshold: Collinearity threshold
            
        Returns:
            List of collinear feature pairs
        """
        corr_matrix = self.compute_correlation_matrix()
        collinear_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                
                if abs(corr_value) >= threshold and not np.isnan(corr_value):
                    collinear_pairs.append((col1, col2, corr_value))
        
        return sorted(collinear_pairs, key=lambda x: -abs(x[2]))
    
    def generate_correlation_heatmap(self, output_file: Optional[str] = None, 
                                   figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        Generate and optionally save correlation heatmap
        
        Args:
            output_file: Path to save the plot
            figsize: Figure size
        """
        corr_matrix = self.compute_correlation_matrix()
        
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(
            corr_matrix, 
            mask=mask,
            annot=True, 
            cmap='coolwarm', 
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={"shrink": .8}
        )
        
        plt.title("Correlation Heatmap", fontsize=16, pad=20)
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_pca_plot(self, output_file: Optional[str] = None, 
                         figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Generate PCA visualization
        
        Args:
            output_file: Path to save the plot
            figsize: Figure size
        """
        if not hasattr(self, 'pca_info'):
            self.compute_multivariate_components(2)
        
        pc_df = self.compute_multivariate_components(2)
        
        plt.figure(figsize=figsize)
        plt.scatter(pc_df['PC1'], pc_df['PC2'], alpha=0.6)
        plt.xlabel(f'PC1 ({self.pca_info["explained_variance"][0]:.1%} variance)')
        plt.ylabel(f'PC2 ({self.pca_info["explained_variance"][1]:.1%} variance)')
        plt.title('Principal Component Analysis', fontsize=16)
        plt.grid(True, alpha=0.3)
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_report(self) -> Dict:
        """
        Generate comprehensive correlation report
        
        Returns:
            Dictionary with all correlation analysis results
        """
        # Basic correlations
        spearman_corr = self.compute_bivariate_correlations('spearman')
        pearson_corr = self.compute_bivariate_correlations('pearson')
        
        # Top correlations
        top_spearman = self.top_correlated_pairs(threshold=0.7, method='spearman')
        top_pearson = self.top_correlated_pairs(threshold=0.7, method='pearson')
        
        # Multivariate analysis
        pc_df = self.compute_multivariate_components(2)
        
        # Feature clusters
        feature_clusters = self.find_feature_clusters(3)
        
        # Collinearity detection
        collinear_features = self.detect_collinearity(threshold=0.9)
        
        report = {
            "bivariate_correlations": {
                "spearman": spearman_corr,
                "pearson": pearson_corr
            },
            "top_correlations": {
                "spearman": top_spearman,
                "pearson": top_pearson
            },
            "multivariate_analysis": {
                "principal_components": pc_df.head().to_dict() if not pc_df.empty else {},
                "pca_info": getattr(self, 'pca_info', {})
            },
            "feature_clusters": feature_clusters,
            "collinear_features": collinear_features,
            "summary": {
                "total_features": len(self.encoded_df.columns),
                "numeric_features": len(self.numeric_df.columns),
                "high_correlations": len(top_spearman),
                "collinear_pairs": len(collinear_features)
            }
        }
        
        return report
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance based on PCA loadings
        
        Returns:
            Dictionary of feature importance scores
        """
        if not hasattr(self, 'pca_info'):
            self.compute_multivariate_components(2)
        
        return self.pca_info.get('feature_importance', {})
    
    def suggest_feature_reduction(self) -> List[str]:
        """
        Suggest features that could be removed due to high collinearity
        
        Returns:
            List of features to consider removing
        """
        collinear_pairs = self.detect_collinearity(threshold=0.95)
        
        # Count how many times each feature appears in collinear pairs
        feature_counts = {}
        for col1, col2, _ in collinear_pairs:
            feature_counts[col1] = feature_counts.get(col1, 0) + 1
            feature_counts[col2] = feature_counts.get(col2, 0) + 1
        
        # Suggest features that appear most frequently in collinear pairs
        suggested_removal = []
        for feature, count in sorted(feature_counts.items(), key=lambda x: x[1], reverse=True):
            if count >= 2:  # Feature appears in at least 2 collinear pairs
                suggested_removal.append(feature)
        
        return suggested_removal[:5]  # Return top 5 suggestions 