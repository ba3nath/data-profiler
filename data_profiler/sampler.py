"""
Data sampling module for large datasets
"""

import pandas as pd
from sqlalchemy import Engine, text
from typing import Optional, Dict, Any
import random


class Sampler:
    """Handles intelligent sampling from large tables"""
    
    def __init__(self, sample_size: int = 10000, random_seed: Optional[int] = None):
        """
        Initialize the sampler
        
        Args:
            sample_size: Number of rows to sample
            random_seed: Random seed for reproducibility
        """
        self.sample_size = sample_size
        if random_seed is not None:
            random.seed(random_seed)
    
    def sample_table(self, engine: Engine, table_name: str, 
                    sample_strategy: str = 'random') -> pd.DataFrame:
        """
        Sample data from a table using various strategies
        
        Args:
            engine: SQLAlchemy engine
            table_name: Name of the table to sample
            sample_strategy: Sampling strategy ('random', 'stratified', 'systematic')
            
        Returns:
            Sampled DataFrame
        """
        if sample_strategy == 'random':
            return self._random_sample(engine, table_name)
        elif sample_strategy == 'stratified':
            return self._stratified_sample(engine, table_name)
        elif sample_strategy == 'systematic':
            return self._systematic_sample(engine, table_name)
        else:
            raise ValueError(f"Unsupported sampling strategy: {sample_strategy}")
    
    def _random_sample(self, engine: Engine, table_name: str) -> pd.DataFrame:
        """Random sampling using SQL RANDOM() or equivalent"""
        # Get total row count
        count_query = f"SELECT COUNT(*) FROM {table_name}"
        with engine.connect() as conn:
            total_rows = conn.execute(text(count_query)).scalar()
        
        if total_rows <= self.sample_size:
            # If table is smaller than sample size, return all rows
            query = f"SELECT * FROM {table_name}"
        else:
            # Use database-specific random sampling
            if 'postgresql' in str(engine.url):
                # PostgreSQL/Redshift
                query = f"""
                    SELECT * FROM {table_name} 
                    ORDER BY RANDOM() 
                    LIMIT {self.sample_size}
                """
            elif 'mysql' in str(engine.url):
                # MySQL
                query = f"""
                    SELECT * FROM {table_name} 
                    ORDER BY RAND() 
                    LIMIT {self.sample_size}
                """
            elif 'bigquery' in str(engine.url):
                # BigQuery
                query = f"""
                    SELECT * FROM {table_name} 
                    ORDER BY RAND() 
                    LIMIT {self.sample_size}
                """
            else:
                # Generic approach
                query = f"""
                    SELECT * FROM {table_name} 
                    LIMIT {self.sample_size}
                """
        
        return pd.read_sql(query, engine)
    
    def _stratified_sample(self, engine: Engine, table_name: str) -> pd.DataFrame:
        """Stratified sampling based on categorical columns"""
        # First, get column information
        columns_query = f"""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = '{table_name}'
        """
        
        with engine.connect() as conn:
            columns = conn.execute(text(columns_query)).fetchall()
        
        # Find categorical columns (string types with limited unique values)
        categorical_cols = []
        for col_name, data_type in columns:
            if data_type in ['varchar', 'text', 'char', 'string']:
                # Check unique count
                unique_query = f"SELECT COUNT(DISTINCT {col_name}) FROM {table_name}"
                unique_count = conn.execute(text(unique_query)).scalar()
                if unique_count <= 50:  # Consider it categorical if <= 50 unique values
                    categorical_cols.append(col_name)
        
        if not categorical_cols:
            # Fall back to random sampling if no categorical columns found
            return self._random_sample(engine, table_name)
        
        # Use the first categorical column for stratification
        strat_col = categorical_cols[0]
        
        # Get distribution of the stratification column
        dist_query = f"""
            SELECT {strat_col}, COUNT(*) as count 
            FROM {table_name} 
            GROUP BY {strat_col}
        """
        
        with engine.connect() as conn:
            distribution = conn.execute(text(dist_query)).fetchall()
        
        # Sample proportionally from each stratum
        sampled_dfs = []
        total_rows = sum(count for _, count in distribution)
        
        for value, count in distribution:
            stratum_size = max(1, int((count / total_rows) * self.sample_size))
            
            if 'postgresql' in str(engine.url):
                stratum_query = f"""
                    SELECT * FROM {table_name} 
                    WHERE {strat_col} = '{value}' 
                    ORDER BY RANDOM() 
                    LIMIT {stratum_size}
                """
            else:
                stratum_query = f"""
                    SELECT * FROM {table_name} 
                    WHERE {strat_col} = '{value}' 
                    LIMIT {stratum_size}
                """
            
            stratum_df = pd.read_sql(stratum_query, engine)
            sampled_dfs.append(stratum_df)
        
        return pd.concat(sampled_dfs, ignore_index=True)
    
    def _systematic_sample(self, engine: Engine, table_name: str) -> pd.DataFrame:
        """Systematic sampling with fixed interval"""
        # Get total row count
        count_query = f"SELECT COUNT(*) FROM {table_name}"
        with engine.connect() as conn:
            total_rows = conn.execute(text(count_query)).scalar()
        
        if total_rows <= self.sample_size:
            return pd.read_sql(f"SELECT * FROM {table_name}", engine)
        
        # Calculate interval
        interval = total_rows // self.sample_size
        
        # Generate systematic sample indices
        sample_indices = list(range(0, total_rows, interval))[:self.sample_size]
        
        # Build query with OFFSET and LIMIT
        sampled_dfs = []
        for i in range(0, len(sample_indices), 100):  # Process in batches
            batch_indices = sample_indices[i:i+100]
            offset_clause = " OR ".join([f"ROW_NUMBER() OVER (ORDER BY 1) = {idx + 1}" for idx in batch_indices])
            
            query = f"""
                SELECT * FROM (
                    SELECT *, ROW_NUMBER() OVER (ORDER BY 1) as rn 
                    FROM {table_name}
                ) t 
                WHERE {offset_clause}
            """
            
            batch_df = pd.read_sql(query, engine)
            sampled_dfs.append(batch_df)
        
        result = pd.concat(sampled_dfs, ignore_index=True)
        return result.drop('rn', axis=1) if 'rn' in result.columns else result
    
    def sample_dataframe(self, df: pd.DataFrame, 
                        sample_strategy: str = 'random') -> pd.DataFrame:
        """
        Sample from a pandas DataFrame
        
        Args:
            df: Input DataFrame
            sample_strategy: Sampling strategy
            
        Returns:
            Sampled DataFrame
        """
        if len(df) <= self.sample_size:
            return df.copy()
        
        if sample_strategy == 'random':
            return df.sample(n=self.sample_size, random_state=42)
        elif sample_strategy == 'stratified':
            return self._stratified_sample_dataframe(df)
        elif sample_strategy == 'systematic':
            return self._systematic_sample_dataframe(df)
        else:
            raise ValueError(f"Unsupported sampling strategy: {sample_strategy}")
    
    def _stratified_sample_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stratified sampling from DataFrame"""
        # Find categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0:
            return df.sample(n=self.sample_size, random_state=42)
        
        # Use the first categorical column for stratification
        strat_col = categorical_cols[0]
        
        # Perform stratified sampling
        sampled_frames = []
        for value in df[strat_col].unique():
            stratum = df[df[strat_col] == value]
            stratum_size = max(1, int((len(stratum) / len(df)) * self.sample_size))
            sampled_stratum = stratum.sample(n=min(stratum_size, len(stratum)), random_state=42)
            sampled_frames.append(sampled_stratum)
        
        return pd.concat(sampled_frames, ignore_index=True)
    
    def _systematic_sample_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Systematic sampling from DataFrame"""
        interval = len(df) // self.sample_size
        indices = list(range(0, len(df), interval))[:self.sample_size]
        return df.iloc[indices].copy() 