import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetPreprocessor:
    """Preprocess EthereumHeist dataset and generate comprehensive overview"""
    
    def __init__(self, dataset_dir: str = "extended_heist_dataset", 
                 output_dir: str = "preprocessed_data"):
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.incidents = []
        self.overview_data = []
        
        # Create output directory
        Path(self.output_dir).mkdir(exist_ok=True)
        
    def scan_dataset(self) -> List[str]:
        """Scan dataset directory for all incident folders"""
        dataset_path = Path(self.dataset_dir)
        
        if not dataset_path.exists():
            logger.error(f"Dataset directory not found: {self.dataset_dir}")
            return []
        
        incidents = [d for d in dataset_path.iterdir() if d.is_dir()]
        self.incidents = sorted([str(d.name) for d in incidents])
        
        logger.info(f"Found {len(self.incidents)} incident folders")
        return self.incidents
    
    def process_incident(self, incident_name: str) -> Dict:
        """Process a single incident and extract statistics"""
        incident_path = Path(self.dataset_dir) / incident_name
        
        result = {
            'incident_name': incident_name,
            'has_accounts': False,
            'has_addresses': False,
            'has_transactions': False,
            'num_hacker_accounts': 0,
            'num_related_addresses': 0,
            'num_transactions': 0,
            'num_normal_tx': 0,
            'num_internal_tx': 0,
            'num_erc20_tx': 0,
            'num_erc721_tx': 0,
            'total_eth_value': 0.0,
            'unique_from_addresses': 0,
            'unique_to_addresses': 0,
            'unique_contracts': 0,
            'earliest_timestamp': None,
            'latest_timestamp': None,
            'year': None,
            'month': None,
            'amount_stolen': None,
            'description': None,
            'data_quality': 'Unknown',
            'temporal_span_days': 0,
            'avg_tx_per_day': 0.0
        }
        
        try:
            # Process accounts-hacker.csv
            accounts_file = incident_path / 'accounts-hacker.csv'
            if accounts_file.exists():
                accounts_df = pd.read_csv(accounts_file)
                result['has_accounts'] = True
                result['num_hacker_accounts'] = len(accounts_df)
                
                if 'year' in accounts_df.columns:
                    result['year'] = accounts_df['year'].iloc[0] if len(accounts_df) > 0 else None
                if 'month' in accounts_df.columns:
                    result['month'] = accounts_df['month'].iloc[0] if len(accounts_df) > 0 else None
                if 'amount_stolen' in accounts_df.columns:
                    result['amount_stolen'] = accounts_df['amount_stolen'].iloc[0] if len(accounts_df) > 0 else None
                if 'description' in accounts_df.columns:
                    result['description'] = accounts_df['description'].iloc[0] if len(accounts_df) > 0 else None
            
            # Process all-address.csv
            addresses_file = incident_path / 'all-address.csv'
            if addresses_file.exists():
                addresses_df = pd.read_csv(addresses_file)
                result['has_addresses'] = True
                result['num_related_addresses'] = len(addresses_df['related_address'].unique()) if 'related_address' in addresses_df.columns else 0
            
            # Process all-tx.csv
            transactions_file = incident_path / 'all-tx.csv'
            if transactions_file.exists():
                tx_df = pd.read_csv(transactions_file)
                result['has_transactions'] = True
                result['num_transactions'] = len(tx_df)
                
                # Transaction type breakdown
                if 'tx_type' in tx_df.columns:
                    result['num_normal_tx'] = len(tx_df[tx_df['tx_type'] == 'normal'])
                    result['num_internal_tx'] = len(tx_df[tx_df['tx_type'] == 'internal'])
                    result['num_erc20_tx'] = len(tx_df[tx_df['tx_type'] == 'erc20'])
                    result['num_erc721_tx'] = len(tx_df[tx_df['tx_type'] == 'erc721'])
                
                # Calculate total ETH value (for normal transactions)
                if 'value' in tx_df.columns and 'tx_type' in tx_df.columns:
                    normal_tx = tx_df[tx_df['tx_type'] == 'normal']
                    try:
                        values = pd.to_numeric(normal_tx['value'], errors='coerce')
                        result['total_eth_value'] = float(values.sum() / 1e18)
                    except:
                        result['total_eth_value'] = 0.0
                
                # Unique addresses
                if 'from' in tx_df.columns:
                    result['unique_from_addresses'] = tx_df['from'].nunique()
                if 'to' in tx_df.columns:
                    result['unique_to_addresses'] = tx_df['to'].nunique()
                if 'contractAddress' in tx_df.columns:
                    result['unique_contracts'] = tx_df['contractAddress'].nunique()
                
                # Timestamp range and temporal features
                if 'timeStamp' in tx_df.columns:
                    timestamps = pd.to_numeric(tx_df['timeStamp'], errors='coerce')
                    timestamps = timestamps.dropna()
                    if len(timestamps) > 0:
                        result['earliest_timestamp'] = int(timestamps.min())
                        result['latest_timestamp'] = int(timestamps.max())
                        
                        # Calculate temporal span
                        span_seconds = result['latest_timestamp'] - result['earliest_timestamp']
                        result['temporal_span_days'] = round(span_seconds / 86400, 2)
                        
                        # Average transactions per day
                        if result['temporal_span_days'] > 0:
                            result['avg_tx_per_day'] = round(result['num_transactions'] / result['temporal_span_days'], 2)
            
            # Load incident summary if exists
            summary_file = incident_path / 'incident_summary.json'
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                    if 'total_unique_addresses' in summary:
                        result['num_related_addresses'] = summary['total_unique_addresses']
            
            # Determine data quality
            result['data_quality'] = self._assess_data_quality(result)
            
        except Exception as e:
            logger.error(f"Error processing incident {incident_name}: {e}")
            result['data_quality'] = 'Error'
        
        return result
    
    def _assess_data_quality(self, result: Dict) -> str:
        """Assess data quality based on completeness"""
        if not result['has_accounts']:
            return 'No Data'
        
        if result['num_transactions'] == 0:
            return 'No Transactions'
        
        if result['num_transactions'] < 10:
            return 'Minimal'
        elif result['num_transactions'] < 100:
            return 'Low'
        elif result['num_transactions'] < 1000:
            return 'Medium'
        else:
            return 'High'
    
    def generate_overview(self, output_file: str = 'dataset_overview.csv'):
        """Generate comprehensive dataset overview CSV"""
        logger.info("Starting dataset overview generation...")
        
        incidents = self.scan_dataset()
        
        if not incidents:
            logger.error("No incidents found to process")
            return
        
        logger.info(f"Processing {len(incidents)} incidents...")
        
        for i, incident_name in enumerate(incidents, 1):
            logger.info(f"Processing {i}/{len(incidents)}: {incident_name}")
            incident_data = self.process_incident(incident_name)
            self.overview_data.append(incident_data)
        
        # Convert to DataFrame
        overview_df = pd.DataFrame(self.overview_data)
        
        # Add computed columns
        overview_df['has_complete_data'] = (
            overview_df['has_accounts'] & 
            overview_df['has_addresses'] & 
            overview_df['has_transactions']
        )
        
        # Convert timestamps to datetime
        if 'earliest_timestamp' in overview_df.columns:
            overview_df['earliest_date'] = pd.to_datetime(
                overview_df['earliest_timestamp'], 
                unit='s', 
                errors='coerce'
            )
            overview_df['latest_date'] = pd.to_datetime(
                overview_df['latest_timestamp'], 
                unit='s', 
                errors='coerce'
            )
        
        # Sort by number of transactions (descending)
        overview_df = overview_df.sort_values('num_transactions', ascending=False)
        
        # Save to output directory
        output_path = Path(self.output_dir) / output_file
        overview_df.to_csv(output_path, index=False)
        logger.info(f"Overview saved to: {output_path}")
        
        # Generate summary statistics
        self._print_summary_stats(overview_df)
        
        return overview_df
    
    def _print_summary_stats(self, df: pd.DataFrame):
        """Print summary statistics"""
        logger.info("\n" + "="*60)
        logger.info("DATASET OVERVIEW SUMMARY")
        logger.info("="*60)
        
        logger.info(f"\nTotal Incidents: {len(df)}")
        logger.info(f"Incidents with Accounts: {df['has_accounts'].sum()}")
        logger.info(f"Incidents with Addresses: {df['has_addresses'].sum()}")
        logger.info(f"Incidents with Transactions: {df['has_transactions'].sum()}")
        logger.info(f"Incidents with Complete Data: {df['has_complete_data'].sum()}")
        
        logger.info(f"\nTotal Hacker Accounts: {df['num_hacker_accounts'].sum()}")
        logger.info(f"Total Related Addresses: {df['num_related_addresses'].sum()}")
        logger.info(f"Total Transactions: {df['num_transactions'].sum()}")
        
        logger.info(f"\nTransaction Breakdown:")
        logger.info(f"  Normal Transactions: {df['num_normal_tx'].sum()}")
        logger.info(f"  Internal Transactions: {df['num_internal_tx'].sum()}")
        logger.info(f"  ERC20 Transactions: {df['num_erc20_tx'].sum()}")
        logger.info(f"  ERC721 Transactions: {df['num_erc721_tx'].sum()}")
        
        logger.info(f"\nTotal ETH Value: {df['total_eth_value'].sum():.2f} ETH")
        
        logger.info(f"\nTemporal Statistics:")
        logger.info(f"  Avg Temporal Span: {df['temporal_span_days'].mean():.2f} days")
        logger.info(f"  Max Temporal Span: {df['temporal_span_days'].max():.2f} days")
        logger.info(f"  Avg Tx per Day: {df['avg_tx_per_day'].mean():.2f}")
        
        logger.info(f"\nData Quality Distribution:")
        quality_counts = df['data_quality'].value_counts()
        for quality, count in quality_counts.items():
            logger.info(f"  {quality}: {count}")
        
        logger.info(f"\nTop 10 Incidents by Transaction Count:")
        top_10 = df.nlargest(10, 'num_transactions')[['incident_name', 'num_transactions', 'data_quality']]
        for idx, row in top_10.iterrows():
            logger.info(f"  {row['incident_name']}: {row['num_transactions']} tx ({row['data_quality']})")
        
        logger.info("\n" + "="*60)
    
    def generate_detailed_report(self, output_file: str = 'dataset_detailed_report.txt'):
        """Generate detailed text report"""
        output_path = Path(self.output_dir) / output_file
        
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ETHEREUM HEIST DATASET - DETAILED REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            for incident_data in self.overview_data:
                f.write(f"\nIncident: {incident_data['incident_name']}\n")
                f.write("-" * 80 + "\n")
                
                if incident_data['year']:
                    f.write(f"Year: {incident_data['year']}\n")
                if incident_data['month']:
                    f.write(f"Month: {incident_data['month']}\n")
                if incident_data['amount_stolen']:
                    f.write(f"Amount Stolen: ${incident_data['amount_stolen']}M\n")
                if incident_data['description']:
                    f.write(f"Description: {incident_data['description']}\n")
                
                f.write(f"\nData Availability:\n")
                f.write(f"  Hacker Accounts: {incident_data['num_hacker_accounts']}\n")
                f.write(f"  Related Addresses: {incident_data['num_related_addresses']}\n")
                f.write(f"  Total Transactions: {incident_data['num_transactions']}\n")
                f.write(f"    - Normal: {incident_data['num_normal_tx']}\n")
                f.write(f"    - Internal: {incident_data['num_internal_tx']}\n")
                f.write(f"    - ERC20: {incident_data['num_erc20_tx']}\n")
                f.write(f"    - ERC721: {incident_data['num_erc721_tx']}\n")
                
                f.write(f"\nNetwork Statistics:\n")
                f.write(f"  Unique From Addresses: {incident_data['unique_from_addresses']}\n")
                f.write(f"  Unique To Addresses: {incident_data['unique_to_addresses']}\n")
                f.write(f"  Unique Contracts: {incident_data['unique_contracts']}\n")
                f.write(f"  Total ETH Value: {incident_data['total_eth_value']:.4f} ETH\n")
                
                f.write(f"\nTemporal Features:\n")
                f.write(f"  Temporal Span: {incident_data['temporal_span_days']} days\n")
                f.write(f"  Avg Tx per Day: {incident_data['avg_tx_per_day']}\n")
                
                if incident_data['earliest_timestamp']:
                    earliest = datetime.fromtimestamp(incident_data['earliest_timestamp'])
                    latest = datetime.fromtimestamp(incident_data['latest_timestamp'])
                    f.write(f"  Earliest: {earliest.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"  Latest: {latest.strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                f.write(f"\nData Quality: {incident_data['data_quality']}\n")
                f.write("\n")
        
        logger.info(f"Detailed report saved to: {output_path}")
    
    def export_graph_ready_data(self, graph_output_dir: str = "graph_ready_data"):
        """Export preprocessed data ready for graph construction"""
        # Create separate output directory for graph data
        graph_path = Path(graph_output_dir)
        graph_path.mkdir(exist_ok=True)
        
        logger.info(f"Exporting graph-ready data to: {graph_path}")
        
        all_nodes = []
        all_edges = []
        temporal_edges = []
        
        for incident_name in self.incidents:
            incident_path = Path(self.dataset_dir) / incident_name
            
            # Load hacker accounts as nodes
            accounts_file = incident_path / 'accounts-hacker.csv'
            if accounts_file.exists():
                accounts_df = pd.read_csv(accounts_file)
                for _, row in accounts_df.iterrows():
                    all_nodes.append({
                        'address': row['address'].lower() if pd.notna(row['address']) else '',
                        'incident_name': incident_name,
                        'node_type': 'hacker',
                        'label': row.get('label', ''),
                        'year': row.get('year', ''),
                        'month': row.get('month', ''),
                        'amount_stolen': row.get('amount_stolen', '')
                    })
            
            # Load all related addresses as nodes
            addresses_file = incident_path / 'all-address.csv'
            if addresses_file.exists():
                addresses_df = pd.read_csv(addresses_file)
                for _, row in addresses_df.iterrows():
                    if 'related_address' in addresses_df.columns:
                        all_nodes.append({
                            'address': row['related_address'].lower() if pd.notna(row['related_address']) else '',
                            'incident_name': incident_name,
                            'node_type': 'related',
                            'label': '',
                            'year': '',
                            'month': '',
                            'amount_stolen': ''
                        })
            
            # Load transactions for edges
            transactions_file = incident_path / 'all-tx.csv'
            if transactions_file.exists():
                tx_df = pd.read_csv(transactions_file)
                
                for idx, tx in tx_df.iterrows():
                    from_addr = str(tx['from']).lower() if pd.notna(tx.get('from')) else ''
                    to_addr = str(tx['to']).lower() if pd.notna(tx.get('to')) else ''
                    
                    if not from_addr or not to_addr:
                        continue
                    
                    # Basic edge data
                    edge_data = {
                        'from': from_addr,
                        'to': to_addr,
                        'incident_name': incident_name,
                        'tx_hash': tx.get('hash', ''),
                        'tx_type': tx.get('tx_type', ''),
                        'value': tx.get('value', 0),
                        'block_number': tx.get('blockNumber', ''),
                        'gas': tx.get('gas', ''),
                        'gas_price': tx.get('gasPrice', ''),
                        'gas_used': tx.get('gasUsed', ''),
                    }
                    
                    # Add token info for ERC20/721
                    if tx.get('tx_type') in ['erc20', 'erc721']:
                        edge_data.update({
                            'contract_address': tx.get('contractAddress', ''),
                            'token_name': tx.get('tokenName', ''),
                            'token_symbol': tx.get('tokenSymbol', ''),
                        })
                    
                    all_edges.append(edge_data)
                    
                    # Temporal edge data (with timestamp features)
                    if pd.notna(tx.get('timeStamp')):
                        timestamp = int(tx['timeStamp'])
                        dt = datetime.fromtimestamp(timestamp)
                        
                        temporal_edge = edge_data.copy()
                        temporal_edge.update({
                            'timestamp': timestamp,
                            'datetime': dt.strftime('%Y-%m-%d %H:%M:%S'),
                            'year': dt.year,
                            'month': dt.month,
                            'day': dt.day,
                            'hour': dt.hour,
                            'day_of_week': dt.weekday(),
                            'is_weekend': 1 if dt.weekday() >= 5 else 0,
                        })
                        temporal_edges.append(temporal_edge)
        
        # Save nodes (remove duplicates)
        nodes_df = pd.DataFrame(all_nodes)
        if len(nodes_df) > 0:
            nodes_df = nodes_df.drop_duplicates(subset=['address'])
            nodes_df = nodes_df[nodes_df['address'] != '']
            nodes_df.to_csv(graph_path / 'nodes.csv', index=False)
            logger.info(f"Exported {len(nodes_df)} unique nodes")
        
        # Save basic edges
        edges_df = pd.DataFrame(all_edges)
        if len(edges_df) > 0:
            edges_df.to_csv(graph_path / 'edges.csv', index=False)
            logger.info(f"Exported {len(edges_df)} edges")
        
        # Save temporal edges
        temporal_df = pd.DataFrame(temporal_edges)
        if len(temporal_df) > 0:
            temporal_df = temporal_df.sort_values('timestamp')
            temporal_df.to_csv(graph_path / 'edges_temporal.csv', index=False)
            logger.info(f"Exported {len(temporal_df)} temporal edges")
        
        # Create metadata file
        metadata = {
            'generation_date': datetime.now().isoformat(),
            'total_incidents': len(self.incidents),
            'total_nodes': len(nodes_df) if len(nodes_df) > 0 else 0,
            'total_edges': len(edges_df) if len(edges_df) > 0 else 0,
            'total_temporal_edges': len(temporal_df) if len(temporal_df) > 0 else 0,
            'files': {
                'nodes.csv': 'All unique addresses (nodes) with metadata',
                'edges.csv': 'All transactions (edges) without temporal info',
                'edges_temporal.csv': 'All transactions with full temporal features'
            },
            'temporal_features': [
                'timestamp', 'datetime', 'year', 'month', 'day', 
                'hour', 'day_of_week', 'is_weekend'
            ],
            'note': 'Use edges_temporal.csv for temporal graph construction'
        }
        
        with open(graph_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Graph-ready data saved to: {graph_path}")
        logger.info(f"Use 'edges_temporal.csv' for temporal graph analysis")

def main():
    """Main execution"""
    
    DATASET_DIR = "extended_heist_dataset"  # Input directory
    OUTPUT_DIR = "preprocessed_data"          # Overview & reports
    GRAPH_DIR = "graph_ready_data"            # Graph construction data
    
    print("\nEthereumHeist Dataset Preprocessor")
    print("="*60)
    
    preprocessor = DatasetPreprocessor(DATASET_DIR, OUTPUT_DIR)
    
    # Generate overview CSV
    print("\n1. Generating dataset overview CSV...")
    overview_df = preprocessor.generate_overview('dataset_overview.csv')
    
    # Generate detailed report
    print("\n2. Generating detailed report...")
    preprocessor.generate_detailed_report('dataset_detailed_report.txt')
    
    # Export graph-ready data (to separate directory)
    print("\n3. Exporting graph-ready data...")
    preprocessor.export_graph_ready_data(GRAPH_DIR)
    
    print("\n" + "="*60)
    print("Preprocessing completed successfully!")
    print("\nOutput Structure:")
    print(f"  {OUTPUT_DIR}/")
    print(f"    ├── dataset_overview.csv       # Dataset statistics")
    print(f"    └── dataset_detailed_report.txt # Detailed report")
    print(f"  {GRAPH_DIR}/")
    print(f"    ├── nodes.csv                  # All unique addresses")
    print(f"    ├── edges.csv                  # All transactions")
    print(f"    ├── edges_temporal.csv         # Transactions with temporal features")
    print(f"    └── metadata.json              # Graph data metadata")
    print("="*60 + "\n")
    print("\nTemporal Features Available:")
    print("  - timestamp (Unix epoch)")
    print("  - datetime (YYYY-MM-DD HH:MM:SS)")
    print("  - year, month, day, hour")
    print("  - day_of_week (0=Monday, 6=Sunday)")
    print("  - is_weekend (0=weekday, 1=weekend)")
    print("\nUse 'edges_temporal.csv' for temporal graph construction!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()