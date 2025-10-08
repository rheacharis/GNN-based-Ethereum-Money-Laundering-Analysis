import pandas as pd
import numpy as np
import networkx as nx
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GraphConstructor:
    """Construct various graph representations from preprocessed data"""
    
    def __init__(self, graph_data_dir: str = "graph_ready_data", 
                 output_dir: str = "constructed_graphs"):
        self.graph_data_dir = Path(graph_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load data
        self.nodes_df = None
        self.edges_df = None
        self.edges_temporal_df = None
        self.metadata = None
        
        # Graphs
        self.static_graph = None
        self.temporal_snapshots = {}
        self.incident_graphs = {}
        
    def load_data(self):
        """Load preprocessed graph data"""
        logger.info("Loading preprocessed data...")
        
        # Load nodes
        nodes_file = self.graph_data_dir / 'nodes.csv'
        if nodes_file.exists():
            self.nodes_df = pd.read_csv(nodes_file)
            logger.info(f"Loaded {len(self.nodes_df)} nodes")
        else:
            logger.error(f"Nodes file not found: {nodes_file}")
            return False
        
        # Load edges
        edges_file = self.graph_data_dir / 'edges.csv'
        if edges_file.exists():
            self.edges_df = pd.read_csv(edges_file)
            logger.info(f"Loaded {len(self.edges_df)} edges")
        
        # Load temporal edges
        temporal_file = self.graph_data_dir / 'edges_temporal.csv'
        if temporal_file.exists():
            self.edges_temporal_df = pd.read_csv(temporal_file)
            logger.info(f"Loaded {len(self.edges_temporal_df)} temporal edges")
        
        # Load metadata
        metadata_file = self.graph_data_dir / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        
        return True
    
    def construct_static_graph(self) -> nx.DiGraph:
        """Construct static directed graph (all transactions, no temporal info)"""
        logger.info("Constructing static directed graph...")
        
        G = nx.DiGraph()
        
        # Add nodes with attributes
        logger.info("Adding nodes...")
        for _, node in self.nodes_df.iterrows():
            G.add_node(
                node['address'],
                node_type=node['node_type'],
                incident=node['incident_name'],
                label=node.get('label', ''),
                year=node.get('year', ''),
                amount_stolen=node.get('amount_stolen', '')
            )
        
        # Add edges with attributes
        logger.info("Adding edges...")
        edge_count = 0
        for _, edge in self.edges_df.iterrows():
            from_addr = edge['from']
            to_addr = edge['to']
            
            # Skip if nodes don't exist
            if from_addr not in G or to_addr not in G:
                continue
            
            # Convert value to float
            try:
                value = float(edge['value'])
            except:
                value = 0.0
            
            # Add edge with attributes
            G.add_edge(
                from_addr,
                to_addr,
                tx_hash=edge.get('tx_hash', ''),
                tx_type=edge.get('tx_type', ''),
                value=value,
                incident=edge.get('incident_name', ''),
                block_number=edge.get('block_number', ''),
                weight=1  # For multi-edge aggregation
            )
            edge_count += 1
        
        self.static_graph = G
        
        logger.info(f"Static graph constructed:")
        logger.info(f"  Nodes: {G.number_of_nodes()}")
        logger.info(f"  Edges: {G.number_of_edges()}")
        logger.info(f"  Density: {nx.density(G):.6f}")
        
        # Save graph
        self._save_graph(G, 'static_graph')
        
        return G
    
    def construct_temporal_graph(self) -> nx.DiGraph:
        """Construct temporal graph with time-ordered edges"""
        logger.info("Constructing temporal directed graph...")
        
        if self.edges_temporal_df is None:
            logger.error("Temporal edges not loaded")
            return None
        
        G = nx.DiGraph()
        
        # Add nodes
        logger.info("Adding nodes...")
        for _, node in self.nodes_df.iterrows():
            G.add_node(
                node['address'],
                node_type=node['node_type'],
                incident=node['incident_name'],
                label=node.get('label', ''),
                year=node.get('year', ''),
                amount_stolen=node.get('amount_stolen', '')
            )
        
        # Sort edges by timestamp
        self.edges_temporal_df = self.edges_temporal_df.sort_values('timestamp')
        
        # Add temporal edges
        logger.info("Adding temporal edges...")
        for idx, edge in self.edges_temporal_df.iterrows():
            from_addr = edge['from']
            to_addr = edge['to']
            
            if from_addr not in G or to_addr not in G:
                continue
            
            try:
                value = float(edge['value'])
            except:
                value = 0.0
            
            # Add edge with temporal attributes
            G.add_edge(
                from_addr,
                to_addr,
                tx_hash=edge.get('tx_hash', ''),
                tx_type=edge.get('tx_type', ''),
                value=value,
                incident=edge.get('incident_name', ''),
                timestamp=edge.get('timestamp', 0),
                datetime=edge.get('datetime', ''),
                year=edge.get('year', ''),
                month=edge.get('month', ''),
                day=edge.get('day', ''),
                hour=edge.get('hour', ''),
                day_of_week=edge.get('day_of_week', ''),
                is_weekend=edge.get('is_weekend', 0),
                block_number=edge.get('block_number', ''),
                weight=1
            )
        
        logger.info(f"Temporal graph constructed:")
        logger.info(f"  Nodes: {G.number_of_nodes()}")
        logger.info(f"  Edges: {G.number_of_edges()}")
        
        # Save graph
        self._save_graph(G, 'temporal_graph')
        
        return G
    
    def construct_incident_graphs(self) -> Dict[str, nx.DiGraph]:
        """Construct separate graphs for each incident"""
        logger.info("Constructing incident-specific graphs...")
        
        incidents = self.nodes_df['incident_name'].unique()
        
        for incident in incidents:
            logger.info(f"Processing incident: {incident}")
            
            # Filter nodes for this incident
            incident_nodes = self.nodes_df[self.nodes_df['incident_name'] == incident]
            incident_addresses = set(incident_nodes['address'].values)
            
            # Filter edges for this incident
            incident_edges = self.edges_df[self.edges_df['incident_name'] == incident]
            
            # Create subgraph
            G = nx.DiGraph()
            
            # Add nodes
            for _, node in incident_nodes.iterrows():
                G.add_node(
                    node['address'],
                    node_type=node['node_type'],
                    incident=node['incident_name'],
                    label=node.get('label', ''),
                    year=node.get('year', ''),
                    amount_stolen=node.get('amount_stolen', '')
                )
            
            # Add edges
            for _, edge in incident_edges.iterrows():
                from_addr = edge['from']
                to_addr = edge['to']
                
                if from_addr not in G or to_addr not in G:
                    continue
                
                try:
                    value = float(edge['value'])
                except:
                    value = 0.0
                
                G.add_edge(
                    from_addr,
                    to_addr,
                    tx_hash=edge.get('tx_hash', ''),
                    tx_type=edge.get('tx_type', ''),
                    value=value,
                    weight=1
                )
            
            self.incident_graphs[incident] = G
            
            logger.info(f"  {incident}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            
            # Save incident graph
            clean_name = self._clean_filename(incident)
            self._save_graph(G, f'incident_{clean_name}')
        
        return self.incident_graphs
    
    def construct_temporal_snapshots(self, time_window: str = 'daily') -> Dict:
        """Construct temporal snapshots of the graph
        
        Args:
            time_window: 'hourly', 'daily', 'weekly', 'monthly'
        """
        logger.info(f"Constructing {time_window} temporal snapshots...")
        
        if self.edges_temporal_df is None:
            logger.error("Temporal edges not loaded")
            return {}
        
        # Group by time window
        if time_window == 'hourly':
            time_key = lambda row: f"{row['year']}-{row['month']:02d}-{row['day']:02d}-{row['hour']:02d}"
        elif time_window == 'daily':
            time_key = lambda row: f"{row['year']}-{row['month']:02d}-{row['day']:02d}"
        elif time_window == 'weekly':
            # Group by ISO week
            time_key = lambda row: pd.Timestamp(row['datetime']).strftime('%Y-W%U')
        elif time_window == 'monthly':
            time_key = lambda row: f"{row['year']}-{row['month']:02d}"
        else:
            logger.error(f"Unknown time window: {time_window}")
            return {}
        
        # Create time-based groups
        self.edges_temporal_df['time_window'] = self.edges_temporal_df.apply(time_key, axis=1)
        time_groups = self.edges_temporal_df.groupby('time_window')
        
        snapshots = {}
        
        for time_period, edges_group in time_groups:
            G = nx.DiGraph()
            
            # Add nodes from these edges
            addresses = set(edges_group['from'].values) | set(edges_group['to'].values)
            
            for addr in addresses:
                node_info = self.nodes_df[self.nodes_df['address'] == addr]
                if len(node_info) > 0:
                    node = node_info.iloc[0]
                    G.add_node(
                        addr,
                        node_type=node['node_type'],
                        incident=node['incident_name'],
                        label=node.get('label', '')
                    )
                else:
                    G.add_node(addr, node_type='unknown', incident='', label='')
            
            # Add edges
            for _, edge in edges_group.iterrows():
                from_addr = edge['from']
                to_addr = edge['to']
                
                if from_addr not in G or to_addr not in G:
                    continue
                
                try:
                    value = float(edge['value'])
                except:
                    value = 0.0
                
                G.add_edge(
                    from_addr,
                    to_addr,
                    value=value,
                    timestamp=edge.get('timestamp', 0),
                    tx_type=edge.get('tx_type', '')
                )
            
            snapshots[time_period] = G
        
        self.temporal_snapshots[time_window] = snapshots
        
        logger.info(f"Created {len(snapshots)} {time_window} snapshots")
        
        # Save snapshots metadata
        snapshot_meta = {
            'time_window': time_window,
            'num_snapshots': len(snapshots),
            'snapshots': {
                period: {
                    'nodes': G.number_of_nodes(),
                    'edges': G.number_of_edges()
                }
                for period, G in snapshots.items()
            }
        }
        
        meta_file = self.output_dir / f'temporal_snapshots_{time_window}_meta.json'
        with open(meta_file, 'w') as f:
            json.dump(snapshot_meta, f, indent=2)
        
        # Save snapshots
        snapshots_dir = self.output_dir / f'snapshots_{time_window}'
        snapshots_dir.mkdir(exist_ok=True)
        
        for period, G in snapshots.items():
            clean_period = self._clean_filename(period)
            snapshot_file = snapshots_dir / f'snapshot_{clean_period}.gpickle'
            with open(snapshot_file, 'wb') as f:
                pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"Snapshots saved to: {snapshots_dir}")
        
        return snapshots
    
    def compute_graph_statistics(self, graph: nx.DiGraph, name: str) -> Dict:
        """Compute comprehensive graph statistics"""
        logger.info(f"Computing statistics for {name}...")
        
        stats = {
            'name': name,
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'density': nx.density(graph),
            'is_connected': nx.is_weakly_connected(graph),
        }
        
        # Degree statistics
        in_degrees = [d for n, d in graph.in_degree()]
        out_degrees = [d for n, d in graph.out_degree()]
        
        stats.update({
            'avg_in_degree': np.mean(in_degrees) if in_degrees else 0,
            'max_in_degree': max(in_degrees) if in_degrees else 0,
            'avg_out_degree': np.mean(out_degrees) if out_degrees else 0,
            'max_out_degree': max(out_degrees) if out_degrees else 0,
        })
        
        # Connected components
        if graph.number_of_nodes() > 0:
            weakly_connected = list(nx.weakly_connected_components(graph))
            strongly_connected = list(nx.strongly_connected_components(graph))
            
            stats.update({
                'num_weakly_connected_components': len(weakly_connected),
                'largest_wcc_size': len(max(weakly_connected, key=len)) if weakly_connected else 0,
                'num_strongly_connected_components': len(strongly_connected),
                'largest_scc_size': len(max(strongly_connected, key=len)) if strongly_connected else 0,
            })
        
        # Node types
        node_types = nx.get_node_attributes(graph, 'node_type')
        type_counts = defaultdict(int)
        for node_type in node_types.values():
            type_counts[node_type] += 1
        stats['node_type_distribution'] = dict(type_counts)
        
        # Edge value statistics
        edge_values = [data.get('value', 0) for _, _, data in graph.edges(data=True)]
        if edge_values:
            stats.update({
                'total_value': sum(edge_values),
                'avg_edge_value': np.mean(edge_values),
                'max_edge_value': max(edge_values),
                'median_edge_value': np.median(edge_values)
            })
        
        return stats
    
    def generate_graph_summary(self):
        """Generate comprehensive summary of all constructed graphs"""
        logger.info("Generating graph construction summary...")
        
        summary = {
            'generation_date': datetime.now().isoformat(),
            'graphs_constructed': [],
            'statistics': {}
        }
        
        # Static graph stats
        if self.static_graph:
            stats = self.compute_graph_statistics(self.static_graph, 'static_graph')
            summary['graphs_constructed'].append('static_graph')
            summary['statistics']['static_graph'] = stats
        
        # Incident graphs stats
        if self.incident_graphs:
            summary['graphs_constructed'].append('incident_graphs')
            summary['statistics']['incident_graphs'] = {}
            
            for incident, G in self.incident_graphs.items():
                stats = self.compute_graph_statistics(G, incident)
                summary['statistics']['incident_graphs'][incident] = stats
        
        # Temporal snapshots stats
        for time_window, snapshots in self.temporal_snapshots.items():
            summary['graphs_constructed'].append(f'temporal_snapshots_{time_window}')
            summary['statistics'][f'temporal_snapshots_{time_window}'] = {
                'num_snapshots': len(snapshots),
                'avg_nodes_per_snapshot': np.mean([G.number_of_nodes() for G in snapshots.values()]),
                'avg_edges_per_snapshot': np.mean([G.number_of_edges() for G in snapshots.values()]),
            }
        
        # Save summary
        summary_file = self.output_dir / 'graph_construction_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary saved to: {summary_file}")
        
        # Print summary
        self._print_summary(summary)
        
        return summary
    
    def _print_summary(self, summary: Dict):
        """Print graph construction summary"""
        logger.info("\n" + "="*60)
        logger.info("GRAPH CONSTRUCTION SUMMARY")
        logger.info("="*60)
        
        logger.info(f"\nGraphs Constructed:")
        for graph_name in summary['graphs_constructed']:
            logger.info(f"  - {graph_name}")
        
        if 'static_graph' in summary['statistics']:
            stats = summary['statistics']['static_graph']
            logger.info(f"\nStatic Graph Statistics:")
            logger.info(f"  Nodes: {stats['num_nodes']}")
            logger.info(f"  Edges: {stats['num_edges']}")
            logger.info(f"  Density: {stats['density']:.6f}")
            logger.info(f"  Avg In-Degree: {stats['avg_in_degree']:.2f}")
            logger.info(f"  Avg Out-Degree: {stats['avg_out_degree']:.2f}")
            if 'total_value' in stats:
                logger.info(f"  Total Value: {stats['total_value']:.2f} Wei")
        
        if 'incident_graphs' in summary['statistics']:
            logger.info(f"\nIncident Graphs: {len(summary['statistics']['incident_graphs'])}")
            
            # Top 5 by size
            incident_stats = summary['statistics']['incident_graphs']
            top_incidents = sorted(
                incident_stats.items(), 
                key=lambda x: x[1]['num_nodes'], 
                reverse=True
            )[:5]
            
            logger.info(f"  Top 5 by nodes:")
            for incident, stats in top_incidents:
                logger.info(f"    {incident}: {stats['num_nodes']} nodes, {stats['num_edges']} edges")
        
        logger.info("\n" + "="*60)
    
    def _save_graph(self, graph: nx.DiGraph, name: str):
        """Save graph in multiple formats"""
        clean_name = self._clean_filename(name)
        
        # Save as pickle (full NetworkX object) - using pickle module directly
        pickle_file = self.output_dir / f'{clean_name}.gpickle'
        try:
            with open(pickle_file, 'wb') as f:
                pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.warning(f"Could not save pickle: {e}")
        
        # Save as GraphML (for Gephi, Cytoscape)
        graphml_file = self.output_dir / f'{clean_name}.graphml'
        try:
            nx.write_graphml(graph, graphml_file)
        except Exception as e:
            logger.warning(f"Could not save GraphML: {e}")
        
        # Save edge list as CSV
        edgelist_file = self.output_dir / f'{clean_name}_edgelist.csv'
        edge_data = []
        for u, v, data in graph.edges(data=True):
            edge_data.append({
                'from': u,
                'to': v,
                **data
            })
        if edge_data:
            pd.DataFrame(edge_data).to_csv(edgelist_file, index=False)
        
        # Save node list as CSV
        nodelist_file = self.output_dir / f'{clean_name}_nodelist.csv'
        node_data = []
        for node, data in graph.nodes(data=True):
            node_data.append({
                'address': node,
                **data
            })
        if node_data:
            pd.DataFrame(node_data).to_csv(nodelist_file, index=False)
        
        logger.info(f"  Saved: {clean_name}")
    
    def _clean_filename(self, name: str) -> str:
        """Clean filename for saving"""
        import re
        clean = re.sub(r'[<>:"/\\|?*]', '_', name)
        clean = re.sub(r'\s+', '_', clean)
        return clean.strip('_')
    
    def construct_all_graphs(self):
        """Construct all graph types"""
        logger.info("Starting full graph construction pipeline...")
        
        # Load data
        if not self.load_data():
            logger.error("Failed to load data")
            return
        
        # Construct static graph
        logger.info("\n" + "="*60)
        self.construct_static_graph()
        
        # Construct temporal graph
        logger.info("\n" + "="*60)
        self.construct_temporal_graph()
        
        # Construct incident-specific graphs
        logger.info("\n" + "="*60)
        self.construct_incident_graphs()
        
        # Construct temporal snapshots
        logger.info("\n" + "="*60)
        self.construct_temporal_snapshots('daily')
        
        logger.info("\n" + "="*60)
        self.construct_temporal_snapshots('monthly')
        
        # Generate summary
        logger.info("\n" + "="*60)
        self.generate_graph_summary()
        
        logger.info("\n" + "="*60)
        logger.info("Graph construction pipeline completed!")
        logger.info("="*60)

def main():
    """Main execution"""
    
    GRAPH_DATA_DIR = "graph_ready_data"
    OUTPUT_DIR = "constructed_graphs"
    
    print("\nEthereumHeist Graph Constructor")
    print("="*60)
    
    constructor = GraphConstructor(GRAPH_DATA_DIR, OUTPUT_DIR)
    
    # Construct all graphs
    constructor.construct_all_graphs()
    
    print("\n" + "="*60)
    print("Graph Construction Completed!")
    print("\nOutput Structure:")
    print(f"  {OUTPUT_DIR}/")
    print(f"    ├── static_graph.gpickle              # Static graph (NetworkX)")
    print(f"    ├── static_graph.graphml              # Static graph (Gephi/Cytoscape)")
    print(f"    ├── temporal_graph.gpickle            # Temporal graph")
    print(f"    ├── incident_*.gpickle                # Per-incident graphs")
    print(f"    ├── snapshots_daily/                  # Daily snapshots")
    print(f"    ├── snapshots_monthly/                # Monthly snapshots")
    print(f"    └── graph_construction_summary.json   # Statistics")
    print("="*60)
    print("\nGraph Types Constructed:")
    print("  1. Static Graph: All transactions, no temporal ordering")
    print("  2. Temporal Graph: Time-ordered edges with temporal features")
    print("  3. Incident Graphs: Separate graph for each heist incident")
    print("  4. Temporal Snapshots: Time-windowed graph snapshots (daily/monthly)")
    print("\nNext Steps:")
    print("  - Use static graph for basic analysis")
    print("  - Use temporal graph for time-aware money flow tracing")
    print("  - Use incident graphs for per-heist analysis")
    print("  - Use snapshots for temporal evolution analysis")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()