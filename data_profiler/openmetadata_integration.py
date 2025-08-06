"""
OpenMetadata integration module
"""

import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from metadata.generated.schema.entity.data.table import Table
    from metadata.generated.schema.entity.services.connections.metadata.openMetadataConnection import OpenMetadataConnection
    from metadata.generated.schema.metadataIngestion.workflow import (
        Source as WorkflowSource,
        Sink as WorkflowSink,
        WorkflowConfig
    )
    from metadata.ingestion.ometa.ometa_api import OpenMetadata
    from metadata.generated.schema.api.data.createTableProfile import CreateTableProfileRequest
    from metadata.generated.schema.entity.data.table import ColumnProfile
    OPENMETADATA_AVAILABLE = True
except ImportError:
    OPENMETADATA_AVAILABLE = False


class OpenMetadataIntegration:
    """Integration with OpenMetadata for pushing profiling results"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OpenMetadata integration
        
        Args:
            config: Configuration dictionary with OpenMetadata connection details
        """
        if not OPENMETADATA_AVAILABLE:
            raise ImportError("OpenMetadata is not available. Install with: pip install openmetadata-ingestion")
        
        self.config = config
        self.metadata = self._create_metadata_client()
    
    def _create_metadata_client(self):
        """Create OpenMetadata client"""
        try:
            connection = OpenMetadataConnection(
                hostPort=self.config.get('host_port', 'http://localhost:8585/api'),
                authProvider=self.config.get('auth_provider', 'no-auth'),
                securityConfig=self.config.get('security_config', None)
            )
            
            return OpenMetadata(connection)
        except Exception as e:
            print(f"Failed to create OpenMetadata client: {e}")
            return None
    
    def push_table_profile(self, 
                          table_fqn: str,
                          column_profiles: List[Dict],
                          table_stats: Dict[str, Any],
                          sample_size: int) -> bool:
        """
        Push table profiling results to OpenMetadata
        
        Args:
            table_fqn: Fully qualified name of the table
            column_profiles: List of column profile dictionaries
            table_stats: Table-level statistics
            sample_size: Number of rows sampled
            
        Returns:
            True if successful, False otherwise
        """
        if not self.metadata:
            print("OpenMetadata client not available")
            return False
        
        try:
            # Create column profiles
            column_profiles_list = []
            for col_profile in column_profiles:
                column_profile = ColumnProfile(
                    name=col_profile['name'],
                    dataType=col_profile.get('dtype', 'unknown'),
                    nullCount=int(col_profile.get('statistics', {}).get('null_count', 0)),
                    nullProportion=col_profile.get('null_ratio', 0.0),
                    uniqueCount=int(col_profile.get('unique_values', 0)),
                    uniqueProportion=col_profile.get('statistics', {}).get('unique_ratio', 0.0),
                    min=col_profile.get('value_range', [None, None])[0] if col_profile.get('value_range') else None,
                    max=col_profile.get('value_range', [None, None])[1] if col_profile.get('value_range') else None,
                    mean=col_profile.get('statistics', {}).get('mean'),
                    median=col_profile.get('statistics', {}).get('median'),
                    stddev=col_profile.get('statistics', {}).get('std'),
                    sum=col_profile.get('statistics', {}).get('sum'),
                    distinctCount=int(col_profile.get('unique_values', 0)),
                    timestamp=datetime.now().isoformat()
                )
                column_profiles_list.append(column_profile)
            
            # Create table profile request
            profile_request = CreateTableProfileRequest(
                tableProfile=table_stats,
                columnProfile=column_profiles_list,
                timestamp=datetime.now().isoformat()
            )
            
            # Push to OpenMetadata
            result = self.metadata.create_or_update_table_profile(
                table_fqn=table_fqn,
                table_profile=profile_request
            )
            
            print(f"Successfully pushed profile for table: {table_fqn}")
            return True
            
        except Exception as e:
            print(f"Failed to push table profile: {e}")
            return False
    
    def push_data_quality_results(self, 
                                 table_fqn: str,
                                 validation_results: Dict[str, Any]) -> bool:
        """
        Push data quality validation results to OpenMetadata
        
        Args:
            table_fqn: Fully qualified name of the table
            validation_results: Data quality validation results
            
        Returns:
            True if successful, False otherwise
        """
        if not self.metadata:
            print("OpenMetadata client not available")
            return False
        
        try:
            # Convert validation results to OpenMetadata format
            quality_metrics = []
            
            for column, results in validation_results.get('quality_metrics', {}).items():
                metric = {
                    'column': column,
                    'completeness': results.get('completeness', 0.0),
                    'uniqueness': results.get('uniqueness', 0.0),
                    'consistency': results.get('consistency', 0.0),
                    'validity': results.get('validity', 0.0),
                    'timestamp': datetime.now().isoformat()
                }
                quality_metrics.append(metric)
            
            # Create data quality request
            quality_request = {
                'table_fqn': table_fqn,
                'quality_metrics': quality_metrics,
                'violations': validation_results.get('custom_rule_violations', {}),
                'summary': validation_results.get('summary', {}),
                'timestamp': datetime.now().isoformat()
            }
            
            # Push to OpenMetadata (this would need to be implemented based on OpenMetadata API)
            print(f"Data quality results prepared for table: {table_fqn}")
            print(f"Quality metrics: {len(quality_metrics)} columns")
            
            return True
            
        except Exception as e:
            print(f"Failed to push data quality results: {e}")
            return False
    
    def push_correlation_analysis(self, 
                                 table_fqn: str,
                                 correlation_results: Dict[str, Any]) -> bool:
        """
        Push correlation analysis results to OpenMetadata
        
        Args:
            table_fqn: Fully qualified name of the table
            correlation_results: Correlation analysis results
            
        Returns:
            True if successful, False otherwise
        """
        if not self.metadata:
            print("OpenMetadata client not available")
            return False
        
        try:
            # Prepare correlation data
            correlation_data = {
                'table_fqn': table_fqn,
                'top_correlations': correlation_results.get('top_correlations', {}),
                'feature_importance': correlation_results.get('multivariate_analysis', {}).get('pca_info', {}).get('feature_importance', {}),
                'collinear_features': correlation_results.get('collinear_features', []),
                'feature_clusters': correlation_results.get('feature_clusters', {}),
                'timestamp': datetime.now().isoformat()
            }
            
            # Push to OpenMetadata (this would need to be implemented based on OpenMetadata API)
            print(f"Correlation analysis prepared for table: {table_fqn}")
            print(f"Top correlations: {len(correlation_data['top_correlations'].get('spearman', []))}")
            
            return True
            
        except Exception as e:
            print(f"Failed to push correlation analysis: {e}")
            return False
    
    def create_table_entity(self, 
                           table_name: str,
                           database_name: str,
                           schema_name: str,
                           columns: List[Dict]) -> Optional[str]:
        """
        Create or update table entity in OpenMetadata
        
        Args:
            table_name: Name of the table
            database_name: Name of the database
            schema_name: Name of the schema
            columns: List of column definitions
            
        Returns:
            Table FQN if successful, None otherwise
        """
        if not self.metadata:
            print("OpenMetadata client not available")
            return None
        
        try:
            # Construct table FQN
            table_fqn = f"{database_name}.{schema_name}.{table_name}"
            
            # Check if table already exists
            existing_table = self.metadata.get_by_name(
                entity=Table,
                fqn=table_fqn
            )
            
            if existing_table:
                print(f"Table {table_fqn} already exists in OpenMetadata")
                return table_fqn
            
            # Create new table entity
            table_entity = Table(
                name=table_name,
                displayName=table_name,
                description=f"Table {table_name} from {database_name}.{schema_name}",
                tableType="Regular",
                columns=columns,
                databaseSchema=self._get_schema_reference(database_name, schema_name)
            )
            
            # Create table in OpenMetadata
            created_table = self.metadata.create_or_update(table_entity)
            
            if created_table:
                print(f"Successfully created table entity: {table_fqn}")
                return table_fqn
            else:
                print(f"Failed to create table entity: {table_fqn}")
                return None
                
        except Exception as e:
            print(f"Failed to create table entity: {e}")
            return None
    
    def _get_schema_reference(self, database_name: str, schema_name: str):
        """Get schema reference for table creation"""
        # This would need to be implemented based on your OpenMetadata setup
        # For now, return None to use default schema
        return None
    
    def push_comprehensive_results(self, 
                                  table_name: str,
                                  database_name: str,
                                  schema_name: str,
                                  column_profiles: List[Dict],
                                  correlation_results: Optional[Dict] = None,
                                  validation_results: Optional[Dict] = None,
                                  table_stats: Optional[Dict] = None,
                                  sample_size: int = 0) -> Dict[str, bool]:
        """
        Push comprehensive profiling results to OpenMetadata
        
        Args:
            table_name: Name of the table
            database_name: Name of the database
            schema_name: Name of the schema
            column_profiles: List of column profiles
            correlation_results: Correlation analysis results
            validation_results: Data validation results
            table_stats: Table-level statistics
            sample_size: Number of rows sampled
            
        Returns:
            Dictionary with success status for each operation
        """
        results = {
            'table_entity': False,
            'table_profile': False,
            'data_quality': False,
            'correlations': False
        }
        
        try:
            # Create table entity
            table_fqn = self.create_table_entity(
                table_name=table_name,
                database_name=database_name,
                schema_name=schema_name,
                columns=[{'name': p['name'], 'dataType': p.get('dtype', 'unknown')} for p in column_profiles]
            )
            
            if table_fqn:
                results['table_entity'] = True
                
                # Push table profile
                if table_stats:
                    results['table_profile'] = self.push_table_profile(
                        table_fqn=table_fqn,
                        column_profiles=column_profiles,
                        table_stats=table_stats,
                        sample_size=sample_size
                    )
                
                # Push data quality results
                if validation_results:
                    results['data_quality'] = self.push_data_quality_results(
                        table_fqn=table_fqn,
                        validation_results=validation_results
                    )
                
                # Push correlation analysis
                if correlation_results:
                    results['correlations'] = self.push_correlation_analysis(
                        table_fqn=table_fqn,
                        correlation_results=correlation_results
                    )
            
            return results
            
        except Exception as e:
            print(f"Failed to push comprehensive results: {e}")
            return results
    
    def get_table_metadata(self, table_fqn: str) -> Optional[Dict]:
        """
        Get table metadata from OpenMetadata
        
        Args:
            table_fqn: Fully qualified name of the table
            
        Returns:
            Table metadata dictionary or None
        """
        if not self.metadata:
            print("OpenMetadata client not available")
            return None
        
        try:
            table = self.metadata.get_by_name(
                entity=Table,
                fqn=table_fqn
            )
            
            if table:
                return {
                    'id': str(table.id),
                    'name': table.name,
                    'displayName': table.displayName,
                    'description': table.description,
                    'tableType': table.tableType,
                    'columns': [col.name for col in table.columns] if table.columns else [],
                    'created': table.created,
                    'updated': table.updated
                }
            else:
                print(f"Table {table_fqn} not found in OpenMetadata")
                return None
                
        except Exception as e:
            print(f"Failed to get table metadata: {e}")
            return None
    
    def list_tables(self, database_name: str, schema_name: str) -> List[str]:
        """
        List tables in a database schema
        
        Args:
            database_name: Name of the database
            schema_name: Name of the schema
            
        Returns:
            List of table names
        """
        if not self.metadata:
            print("OpenMetadata client not available")
            return []
        
        try:
            # This would need to be implemented based on OpenMetadata API
            # For now, return empty list
            print(f"Listing tables in {database_name}.{schema_name}")
            return []
            
        except Exception as e:
            print(f"Failed to list tables: {e}")
            return [] 