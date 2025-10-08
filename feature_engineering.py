import pandas as pd
import numpy as np
import networkx as nx
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Extract features from constructed graphs for ML models"""
    
    def __init__(self, graphs_dir: str = "constructed_graphs",
                 graph_data_dir: str = "graph_ready_data",
                 output_dir: str = "extracted_features"):
        self.graphs_dir = Path(graphs_dir)
        self.graph_data_dir = Path(graph_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load graphs
        self.static_graph = None
        self.temporal_graph = None
        
        # Load raw data
        self.nodes_df = None
        self.edges_temporal_df = None
        
        # Feature storage
        self.node_features = {}
        self.edge_features = {}
        
    def load_graphs(self):
        """Load constructed graphs"""
        logger.info("Loading graphs...")
        
        # Load static graph
        static_file = self.graphs_dir / 'static_graph.gpickle'
        if static_file.exists():
            with open(static_file, 'rb') as f:
                self.static_graph = pickle.load(f)
            logger.info(f"Loaded static graph: {self.static_graph.number_of_nodes()} nodes, {self.static_graph.number_of_edges()} edges")
        else:
            logger.error(f"Static graph not found: {static_file}")
            return False
        
        # Load temporal graph
        temporal_file = self.graphs_dir / 'temporal_graph.gpickle'
        if temporal_file.exists():
            with open(temporal_file, 'rb') as f:
                self.temporal_graph = pickle.load(f)
            logger.info(f"Loaded temporal graph: {self.temporal_graph.number_of_nodes()} nodes")
        
        return True
    
    def load_raw_data(self):
        """Load raw preprocessed data"""
        logger.info("Loading raw data...")
        
        # Load nodes
        nodes_file = self.graph_data_dir / 'nodes.csv'
        if nodes_file.exists():
            self.nodes_df = pd.read_csv(nodes_file)
            logger.info(f"Loaded {len(self.nodes_df)} nodes")
        
        # Load temporal edges
        temporal_edges_file = self.graph_data_dir / 'edges_temporal.csv'
        if temporal_edges_file.exists():
            self.edges_temporal_df = pd.read_csv(temporal_edges_file)
            logger.info(f"Loaded {len(self.edges_temporal_df)} temporal edges")
        
        return True
    
    def extract_basic_node_features(self) -> pd.DataFrame:
        """Extract basic node-level features"""
        logger.info("Extracting basic node features...")
        
        features = []
        
        for node in self.static_graph.nodes():
            node_data = self.static_graph.nodes[node]
            
            # Degree features
            in_degree = self.static_graph.in_degree(node)
            out_degree = self.static_graph.out_degree(node)
            total_degree = in_degree + out_degree
            
            # Get in/out edges
            in_edges = list(self.static_graph.in_edges(node, data=True))
            out_edges = list(self.static_graph.out_edges(node, data=True))
            
            # Value features
            in_values = [data.get('value', 0) for _, _, data in in_edges]
            out_values = [data.get('value', 0) for _, _, data in out_edges]
            
            total_in_value = sum(in_values)
            total_out_value = sum(out_values)
            
            # Balance
            balance = total_in_value - total_out_value
            
            feature = {
                'address': node,
                'node_type': node_data.get('node_type', 'unknown'),
                'incident': node_data.get('incident', ''),
                'label': node_data.get('label', ''),
                
                # Degree features
                'in_degree': in_degree,
                'out_degree': out_degree,
                'total_degree': total_degree,
                'degree_ratio': out_degree / in_degree if in_degree > 0 else 0,
                
                # Value features (in Wei)
                'total_in_value': total_in_value,
                'total_out_value': total_out_value,
                'total_value': total_in_value + total_out_value,
                'balance': balance,
                'balance_ratio': balance / total_in_value if total_in_value > 0 else 0,
                
                # Average transaction values
                'avg_in_value': np.mean(in_values) if in_values else 0,
                'avg_out_value': np.mean(out_values) if out_values else 0,
                'max_in_value': max(in_values) if in_values else 0,
                'max_out_value': max(out_values) if out_values else 0,
                'min_in_value': min(in_values) if in_values else 0,
                'min_out_value': min(out_values) if out_values else 0,
                
                # Transaction count features
                'num_in_tx': len(in_edges),
                'num_out_tx': len(out_edges),
                'total_tx': len(in_edges) + len(out_edges),
                
                # Unique counterparties
                'unique_senders': len(set([u for u, _, _ in in_edges])),
                'unique_receivers': len(set([v for _, v, _ in out_edges])),
                'unique_counterparties': len(set([u for u, _, _ in in_edges] + [v for _, v, _ in out_edges])),
            }
            
            features.append(feature)
        
        features_df = pd.DataFrame(features)
        logger.info(f"Extracted {len(features_df)} node features with {len(features_df.columns)} columns")
        
        return features_df
    
    def extract_temporal_node_features(self) -> pd.DataFrame:
        """Extract temporal features for nodes"""
        logger.info("Extracting temporal node features...")
        
        if self.edges_temporal_df is None:
            logger.warning("Temporal edges not loaded, skipping temporal features")
            return pd.DataFrame()
        
        features = []
        
        # Group by address (both from and to)
        for node in self.static_graph.nodes():
            # Get all transactions involving this node
            node_tx = self.edges_temporal_df[
                (self.edges_temporal_df['from'] == node) | 
                (self.edges_temporal_df['to'] == node)
            ].copy()
            
            if len(node_tx) == 0:
                continue
            
            # Sort by timestamp
            node_tx = node_tx.sort_values('timestamp')
            
            # Temporal statistics
            timestamps = node_tx['timestamp'].values
            
            feature = {
                'address': node,
                
                # Activity period
                'first_tx_timestamp': int(timestamps[0]) if len(timestamps) > 0 else 0,
                'last_tx_timestamp': int(timestamps[-1]) if len(timestamps) > 0 else 0,
                'activity_duration_seconds': int(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0,
                'activity_duration_days': (timestamps[-1] - timestamps[0]) / 86400 if len(timestamps) > 1 else 0,
                
                # Activity frequency
                'num_active_days': node_tx['day'].nunique() if 'day' in node_tx.columns else 0,
                'num_active_months': node_tx['month'].nunique() if 'month' in node_tx.columns else 0,
                'avg_tx_per_day': len(node_tx) / ((timestamps[-1] - timestamps[0]) / 86400) if len(timestamps) > 1 and (timestamps[-1] - timestamps[0]) > 0 else 0,
                
                # Time patterns
                'weekend_tx_count': int(node_tx['is_weekend'].sum()) if 'is_weekend' in node_tx.columns else 0,
                'weekday_tx_count': int((1 - node_tx['is_weekend']).sum()) if 'is_weekend' in node_tx.columns else 0,
                'weekend_tx_ratio': node_tx['is_weekend'].mean() if 'is_weekend' in node_tx.columns else 0,
                
                # Hourly patterns
                'most_active_hour': int(node_tx['hour'].mode()[0]) if 'hour' in node_tx.columns and len(node_tx) > 0 else -1,
                'night_tx_count': int(node_tx[node_tx['hour'].isin([0,1,2,3,4,5])]['hour'].count()) if 'hour' in node_tx.columns else 0,
                
                # Transaction velocity
                'avg_time_between_tx': np.mean(np.diff(timestamps)) if len(timestamps) > 1 else 0,
                'min_time_between_tx': np.min(np.diff(timestamps)) if len(timestamps) > 1 else 0,
                'max_time_between_tx': np.max(np.diff(timestamps)) if len(timestamps) > 1 else 0,
            }
            
            features.append(feature)
        
        features_df = pd.DataFrame(features)
        logger.info(f"Extracted temporal features for {len(features_df)} nodes")
        
        return features_df
    
    def extract_network_position_features(self) -> pd.DataFrame:
        """Extract graph centrality and position features"""
        logger.info("Extracting network position features...")
        
        features = []
        
        # Calculate centrality metrics (can be slow for large graphs)
        logger.info("  Computing PageRank...")
        pagerank = nx.pagerank(self.static_graph, max_iter=100)
        
        logger.info("  Computing betweenness centrality (sample)...")
        # Use k parameter for large graphs to sample
        num_nodes = self.static_graph.number_of_nodes()
        k = min(100, num_nodes)  # Sample 100 nodes for betweenness
        betweenness = nx.betweenness_centrality(self.static_graph, k=k)
        
        logger.info("  Computing closeness centrality...")
        # Compute only for strongly connected component
        try:
            largest_scc = max(nx.strongly_connected_components(self.static_graph), key=len)
            scc_graph = self.static_graph.subgraph(largest_scc)
            closeness = nx.closeness_centrality(scc_graph)
        except:
            closeness = {}
        
        logger.info("  Computing clustering coefficient...")
        # For directed graphs
        clustering = nx.clustering(self.static_graph.to_undirected())
        
        for node in self.static_graph.nodes():
            feature = {
                'address': node,
                'pagerank': pagerank.get(node, 0),
                'betweenness_centrality': betweenness.get(node, 0),
                'closeness_centrality': closeness.get(node, 0),
                'clustering_coefficient': clustering.get(node, 0),
            }
            
            features.append(feature)
        
        features_df = pd.DataFrame(features)
        logger.info(f"Extracted network position features for {len(features_df)} nodes")
        
        return features_df
    
    def extract_behavioral_features(self) -> pd.DataFrame:
        """Extract behavioral pattern features"""
        logger.info("Extracting behavioral features...")
        
        features = []
        
        for node in self.static_graph.nodes():
            # Get neighbors
            predecessors = list(self.static_graph.predecessors(node))
            successors = list(self.static_graph.successors(node))
            
            # Fan-in / Fan-out patterns
            fan_in = len(predecessors)
            fan_out = len(successors)
            
            # Check for common patterns
            in_edges = list(self.static_graph.in_edges(node, data=True))
            out_edges = list(self.static_graph.out_edges(node, data=True))
            
            # Get transaction values
            in_values = [data.get('value', 0) for _, _, data in in_edges]
            out_values = [data.get('value', 0) for _, _, data in out_edges]
            
            # Detect patterns
            # 1. Peeling chain: many small outputs
            if len(out_values) > 0:
                median_out = np.median(out_values)
                num_small_outputs = sum(1 for v in out_values if v < median_out)
            else:
                num_small_outputs = 0
            
            # 2. Mixing behavior: many inputs, many outputs
            is_potential_mixer = fan_in > 10 and fan_out > 10
            
            # 3. One-time address: single in, single out
            is_one_time = fan_in == 1 and fan_out == 1
            
            # 4. Collection address: many in, few out
            is_collector = fan_in > 5 and fan_out <= 2
            
            # 5. Distribution address: few in, many out
            is_distributor = fan_in <= 2 and fan_out > 5
            
            # Value distribution
            in_value_std = np.std(in_values) if len(in_values) > 1 else 0
            out_value_std = np.std(out_values) if len(out_values) > 1 else 0
            
            feature = {
                'address': node,
                'fan_in': fan_in,
                'fan_out': fan_out,
                'fan_ratio': fan_out / fan_in if fan_in > 0 else 0,
                
                # Behavioral flags
                'is_potential_mixer': int(is_potential_mixer),
                'is_one_time_address': int(is_one_time),
                'is_collector': int(is_collector),
                'is_distributor': int(is_distributor),
                
                # Value patterns
                'in_value_std': in_value_std,
                'out_value_std': out_value_std,
                'num_small_outputs': num_small_outputs,
                'small_output_ratio': num_small_outputs / len(out_values) if len(out_values) > 0 else 0,
                
                # Interaction patterns
                'self_loop': int(self.static_graph.has_edge(node, node)),
                'reciprocal_edges': sum(1 for succ in successors if self.static_graph.has_edge(succ, node)),
            }
            
            features.append(feature)
        
        features_df = pd.DataFrame(features)
        logger.info(f"Extracted behavioral features for {len(features_df)} nodes")
        
        return features_df
    
    def extract_edge_features(self) -> pd.DataFrame:
        """Extract edge-level features"""
        logger.info("Extracting edge features...")
        
        if self.edges_temporal_df is None:
            logger.warning("Temporal edges not loaded, using basic edges")
            return pd.DataFrame()
        
        features = []
        
        # Process each edge
        for idx, edge in self.edges_temporal_df.iterrows():
            from_addr = edge['from']
            to_addr = edge['to']
            
            # Basic edge features
            try:
                value = float(edge.get('value', 0))
            except:
                value = 0.0
            
            feature = {
                'from': from_addr,
                'to': to_addr,
                'tx_hash': edge.get('tx_hash', ''),
                'incident': edge.get('incident_name', ''),
                
                # Value features
                'value': value,
                'value_eth': value / 1e18,
                'log_value': np.log1p(value),
                
                # Temporal features
                'timestamp': edge.get('timestamp', 0),
                'hour': edge.get('hour', -1),
                'day_of_week': edge.get('day_of_week', -1),
                'is_weekend': edge.get('is_weekend', 0),
                
                # Transaction type
                'tx_type': edge.get('tx_type', ''),
                'is_erc20': int(edge.get('tx_type') == 'erc20'),
                'is_internal': int(edge.get('tx_type') == 'internal'),
            }
            
            features.append(feature)
            
            # Limit to avoid memory issues
            if idx > 0 and idx % 100000 == 0:
                logger.info(f"  Processed {idx} edges...")
        
        features_df = pd.DataFrame(features)
        logger.info(f"Extracted features for {len(features_df)} edges")
        
        return features_df
    
    def merge_all_node_features(self) -> pd.DataFrame:
        """Merge all node feature DataFrames"""
        logger.info("Merging all node features...")
        
        # Extract all feature types
        basic_features = self.extract_basic_node_features()
        temporal_features = self.extract_temporal_node_features()
        network_features = self.extract_network_position_features()
        behavioral_features = self.extract_behavioral_features()
        
        # Merge on address
        merged = basic_features
        
        if len(temporal_features) > 0:
            merged = merged.merge(temporal_features, on='address', how='left')
        
        if len(network_features) > 0:
            merged = merged.merge(network_features, on='address', how='left')
        
        if len(behavioral_features) > 0:
            merged = merged.merge(behavioral_features, on='address', how='left')
        
        # Fill NaN values
        merged = merged.fillna(0)
        
        # Add label column (1 for hacker, 0 for others)
        merged['is_hacker'] = (merged['node_type'] == 'hacker').astype(int)
        
        logger.info(f"Final merged features: {len(merged)} nodes, {len(merged.columns)} features")
        
        return merged
    
    def generate_feature_summary(self, features_df: pd.DataFrame) -> Dict:
        """Generate feature summary statistics"""
        logger.info("Generating feature summary...")
        
        summary = {
            'generation_date': datetime.now().isoformat(),
            'num_samples': len(features_df),
            'num_features': len(features_df.columns),
            'feature_columns': list(features_df.columns),
            'label_distribution': {},
            'feature_statistics': {}
        }
        
        # Label distribution
        if 'is_hacker' in features_df.columns:
            summary['label_distribution'] = {
                'hacker': int(features_df['is_hacker'].sum()),
                'non_hacker': int((1 - features_df['is_hacker']).sum()),
                'hacker_ratio': float(features_df['is_hacker'].mean())
            }
        
        # Feature statistics
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in ['is_hacker']:
                continue
            
            summary['feature_statistics'][col] = {
                'mean': float(features_df[col].mean()),
                'std': float(features_df[col].std()),
                'min': float(features_df[col].min()),
                'max': float(features_df[col].max()),
                'median': float(features_df[col].median()),
                'non_zero_ratio': float((features_df[col] != 0).mean())
            }
        
        return summary
    
    def save_features(self, features_df: pd.DataFrame, name: str):
        """Save features to files"""
        logger.info(f"Saving {name} features...")
        
        # Save as CSV
        csv_file = self.output_dir / f'{name}_features.csv'
        features_df.to_csv(csv_file, index=False)
        logger.info(f"  Saved CSV: {csv_file}")
        
        # Save as numpy arrays (for ML models)
        if 'is_hacker' in features_df.columns:
            # Separate features and labels
            feature_cols = [col for col in features_df.columns 
                          if col not in ['address', 'node_type', 'incident', 'label', 'is_hacker']]
            
            X = features_df[feature_cols].values
            y = features_df['is_hacker'].values
            addresses = features_df['address'].values
            
            np.save(self.output_dir / f'{name}_X.npy', X)
            np.save(self.output_dir / f'{name}_y.npy', y)
            np.save(self.output_dir / f'{name}_addresses.npy', addresses)
            
            # Save feature names
            with open(self.output_dir / f'{name}_feature_names.json', 'w') as f:
                json.dump(feature_cols, f, indent=2)
            
            logger.info(f"  Saved numpy arrays: X shape {X.shape}, y shape {y.shape}")
        
        # Generate and save summary
        summary = self.generate_feature_summary(features_df)
        summary_file = self.output_dir / f'{name}_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"  Saved summary: {summary_file}")
    
    def extract_all_features(self):
        """Extract all features"""
        logger.info("Starting feature extraction pipeline...")
        logger.info("="*60)
        
        # Load graphs and data
        if not self.load_graphs():
            logger.error("Failed to load graphs")
            return
        
        self.load_raw_data()
        
        # Extract node features
        logger.info("\n" + "="*60)
        logger.info("EXTRACTING NODE FEATURES")
        logger.info("="*60)
        node_features = self.merge_all_node_features()
        self.save_features(node_features, 'node')
        
        # Extract edge features
        logger.info("\n" + "="*60)
        logger.info("EXTRACTING EDGE FEATURES")
        logger.info("="*60)
        edge_features = self.extract_edge_features()
        if len(edge_features) > 0:
            self.save_features(edge_features, 'edge')
        
        logger.info("\n" + "="*60)
        logger.info("FEATURE EXTRACTION COMPLETED!")
        logger.info("="*60)
        
        # Print summary
        self._print_extraction_summary(node_features, edge_features)
    
    def _print_extraction_summary(self, node_features: pd.DataFrame, edge_features: pd.DataFrame):
        """Print extraction summary"""
        logger.info("\n" + "="*60)
        logger.info("FEATURE EXTRACTION SUMMARY")
        logger.info("="*60)
        
        logger.info(f"\nNode Features:")
        logger.info(f"  Total Nodes: {len(node_features)}")
        logger.info(f"  Total Features: {len(node_features.columns)}")
        logger.info(f"  Hacker Nodes: {node_features['is_hacker'].sum()}")
        logger.info(f"  Non-Hacker Nodes: {(1 - node_features['is_hacker']).sum()}")
        logger.info(f"  Class Imbalance Ratio: {node_features['is_hacker'].mean():.4f}")
        
        logger.info(f"\nFeature Categories:")
        basic_cols = ['in_degree', 'out_degree', 'total_value', 'balance']
        temporal_cols = ['activity_duration_days', 'avg_tx_per_day', 'weekend_tx_ratio']
        network_cols = ['pagerank', 'betweenness_centrality', 'clustering_coefficient']
        behavioral_cols = ['is_potential_mixer', 'is_collector', 'is_distributor']
        
        logger.info(f"  Basic: {sum(1 for c in basic_cols if c in node_features.columns)} features")
        logger.info(f"  Temporal: {sum(1 for c in temporal_cols if c in node_features.columns)} features")
        logger.info(f"  Network: {sum(1 for c in network_cols if c in node_features.columns)} features")
        logger.info(f"  Behavioral: {sum(1 for c in behavioral_cols if c in node_features.columns)} features")
        
        if len(edge_features) > 0:
            logger.info(f"\nEdge Features:")
            logger.info(f"  Total Edges: {len(edge_features)}")
            logger.info(f"  Total Features: {len(edge_features.columns)}")
        
        logger.info("\n" + "="*60)

def main():
    """Main execution"""
    
    GRAPHS_DIR = "constructed_graphs"
    GRAPH_DATA_DIR = "graph_ready_data"
    OUTPUT_DIR = "extracted_features"
    
    print("\nEthereumHeist Feature Engineering")
    print("="*60)
    
    engineer = FeatureEngineer(GRAPHS_DIR, GRAPH_DATA_DIR, OUTPUT_DIR)
    engineer.extract_all_features()
    
    print("\n" + "="*60)
    print("Feature Extraction Completed!")
    print("\nOutput Structure:")
    print(f"  {OUTPUT_DIR}/")
    print(f"    ├── node_features.csv           # All node features (CSV)")
    print(f"    ├── node_X.npy                  # Feature matrix (numpy)")
    print(f"    ├── node_y.npy                  # Labels (numpy)")
    print(f"    ├── node_addresses.npy          # Address mapping")
    print(f"    ├── node_feature_names.json     # Feature names")
    print(f"    ├── node_summary.json           # Statistics")
    print(f"    ├── edge_features.csv           # Edge features")
    print(f"    └── edge_summary.json           # Edge statistics")
    print("="*60)
    print("\nFeature Categories Extracted:")
    print("  1. Basic: degree, values, balances, transaction counts")
    print("  2. Temporal: activity patterns, time-based features")
    print("  3. Network: centrality, clustering, position")
    print("  4. Behavioral: mixer detection, patterns, flags")
    print("\nNext Steps:")
    print("  - Use node_X.npy and node_y.npy for ML model training")
    print("  - Perform feature selection/importance analysis")
    print("  - Train classifiers (Phase 6)")
    print("  - Apply clustering algorithms (Phase 5)")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()