"""
Graph SMOTE - Synthetic Minority Oversampling for Graph Data
Optimized version (numeric-only feature space for neighbor search)
"""

import numpy as np
import pandas as pd
import networkx as nx
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
from sklearn.neighbors import NearestNeighbors
import logging
import json

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GraphSMOTE:
    """
    Graph-aware SMOTE for balancing phase-labeled transaction graphs
    """

    def __init__(self, k_neighbors: int = 5, target_ratio: float = 0.5):
        """
        Args:
            k_neighbors: Number of neighbors for SMOTE synthesis
            target_ratio: Target minority class ratio (e.g., 0.5 = 50% of majority)
        """
        self.k_neighbors = k_neighbors
        self.target_ratio = target_ratio

    def fit_resample(self, G: nx.DiGraph, features_df: pd.DataFrame,
                     phase_column: str = 'phase') -> Tuple[nx.DiGraph, pd.DataFrame]:
        """
        Apply optimized Graph SMOTE using numeric feature space neighbor search
        Returns:
            Balanced graph and features dataframe
        """
        logger.info("Starting optimized Graph SMOTE...")

        # Get phase distribution
        phase_counts = features_df[phase_column].value_counts()
        logger.info(f"Original distribution:\n{phase_counts}")

        # Identify majority and minority classes
        majority_class = phase_counts.idxmax()
        majority_count = phase_counts.max()

        synthetic_nodes = []
        G_balanced = G.copy()

        # Keep only numeric columns for neighbor computation
        numeric_df = features_df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise ValueError("No numeric columns found in features_df for neighbor computation.")

        for phase, count in phase_counts.items():
            if phase == majority_class or phase == 'unknown':
                continue

            # Calculate target count (balance to target_ratio of majority)
            target_count = int(majority_count * self.target_ratio)
            n_synthetic = max(0, target_count - count)

            if n_synthetic == 0:
                continue

            logger.info(f"Generating {n_synthetic} synthetic nodes for phase: {phase}")

            # Get minority class nodes
            minority_nodes = features_df[features_df[phase_column] == phase].index.tolist()

            # Fit NearestNeighbors on numeric subset
            X_minority = numeric_df.loc[minority_nodes].values
            nbrs = NearestNeighbors(
                n_neighbors=min(self.k_neighbors, len(minority_nodes) - 1),
                algorithm='ball_tree'
            ).fit(X_minority)

            # Generate synthetic samples
            for i in range(n_synthetic):
                synthetic_node, synthetic_features = self._create_synthetic_node_fast(
                    G, features_df, numeric_df, minority_nodes, nbrs, phase, i
                )

                # Add synthetic node to graph
                G_balanced.add_node(synthetic_node, **synthetic_features)

                # Add synthetic edges (sampled within minority community)
                self._add_synthetic_edges(G_balanced, synthetic_node, minority_nodes)

                # Record synthetic node metadata
                synthetic_nodes.append({
                    'address': synthetic_node,
                    phase_column: phase,
                    'is_synthetic': True,
                    **synthetic_features
                })

        # Create balanced features dataframe
        if synthetic_nodes:
            synthetic_df = pd.DataFrame(synthetic_nodes)
            synthetic_df.set_index('address', inplace=True)
            features_df['is_synthetic'] = False
            balanced_df = pd.concat([features_df, synthetic_df], axis=0)
        else:
            balanced_df = features_df.copy()
            balanced_df['is_synthetic'] = False

        # Report results
        balanced_counts = balanced_df[phase_column].value_counts()
        logger.info(f"Balanced distribution:\n{balanced_counts}")

        return G_balanced, balanced_df

    def _create_synthetic_node_fast(self, G, features_df, numeric_df, minority_nodes,
                                    nbrs, phase, synthetic_id):
        """
        Faster synthetic node generation using numeric feature interpolation
        """
        # Choose random anchor
        anchor_idx = np.random.randint(0, len(minority_nodes))
        anchor_node = minority_nodes[anchor_idx]

        # Find neighbors in numeric space
        distances, indices = nbrs.kneighbors([numeric_df.loc[anchor_node].values])
        neighbors = [minority_nodes[i] for i in indices[0] if minority_nodes[i] != anchor_node]

        if not neighbors:
            neighbor_node = np.random.choice(minority_nodes)
        else:
            neighbor_node = np.random.choice(neighbors)

        anchor_features = features_df.loc[anchor_node]
        neighbor_features = features_df.loc[neighbor_node]

        # SMOTE interpolation
        lambda_val = np.random.uniform(0, 1)
        synthetic_features = {}

        for col in features_df.columns:
            if col in ['phase', 'address', 'is_hacker_seed', 'is_synthetic']:
                continue
            try:
                anchor_val = float(anchor_features[col])
                neighbor_val = float(neighbor_features[col])
                synthetic_val = anchor_val + lambda_val * (neighbor_val - anchor_val)
                synthetic_features[col] = synthetic_val
            except (ValueError, TypeError):
                synthetic_features[col] = anchor_features[col]

        # New synthetic node ID
        synthetic_node_id = f"synthetic_{phase}_{synthetic_id}"

        # Add meta
        synthetic_features.update({
            'node_type': 'synthetic',
            'phase': phase,
            'is_synthetic': True,
            'anchor_node': anchor_node,
            'neighbor_node': neighbor_node
        })

        return synthetic_node_id, synthetic_features

    def _add_synthetic_edges(self, G: nx.DiGraph, synthetic_node: str,
                             minority_nodes: List[str]):
        """
        Add edges for synthetic node based on minority connections
        """
        num_connections = min(np.random.randint(1, 4), len(minority_nodes))
        connected_nodes = np.random.choice(minority_nodes, size=num_connections, replace=False)

        for neighbor in connected_nodes:
            if np.random.rand() > 0.5:
                G.add_edge(synthetic_node, neighbor,
                           value=np.random.exponential(0.5),
                           is_synthetic=True)
            else:
                G.add_edge(neighbor, synthetic_node,
                           value=np.random.exponential(0.5),
                           is_synthetic=True)

def main():
    """Apply optimized Graph SMOTE to phase-labeled data"""

    # Paths
    PHASE_FEATURES_PATH = 'phase_classification/phase_labeled_features_behavioral.csv'
    GRAPH_PATH = 'constructed_graphs/static_graph.gpickle'
    OUTPUT_DIR = Path('balanced_data')
    OUTPUT_DIR.mkdir(exist_ok=True)

    logger.info("Loading phase-labeled data...")

    # Load features
    features_df = pd.read_csv(PHASE_FEATURES_PATH)
    features_df.set_index('address', inplace=True)

    # Load graph
    with open(GRAPH_PATH, 'rb') as f:
        G = pickle.load(f)

    logger.info(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    logger.info(f"Loaded features: {len(features_df)} addresses")

    # Initialize SMOTE
    smote = GraphSMOTE(k_neighbors=5, target_ratio=0.5)

    # Apply balancing
    G_balanced, features_balanced = smote.fit_resample(G, features_df, phase_column='phase')

    # Save results
    logger.info("Saving balanced data...")
    features_balanced.to_csv(OUTPUT_DIR / 'balanced_features.csv')

    with open(OUTPUT_DIR / 'balanced_graph.gpickle', 'wb') as f:
        pickle.dump(G_balanced, f)

    metadata = {
        'original_nodes': G.number_of_nodes(),
        'balanced_nodes': G_balanced.number_of_nodes(),
        'original_edges': G.number_of_edges(),
        'balanced_edges': G_balanced.number_of_edges(),
        'original_distribution': features_df['phase'].value_counts().to_dict(),
        'balanced_distribution': features_balanced['phase'].value_counts().to_dict(),
        'synthetic_nodes': int(features_balanced['is_synthetic'].sum()),
        'k_neighbors': smote.k_neighbors,
        'target_ratio': smote.target_ratio
    }

    with open(OUTPUT_DIR / 'balancing_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("Graph SMOTE complete!")
    logger.info(f"Balanced data saved to: {OUTPUT_DIR}")

    # Summary print
    print("\n" + "=" * 60)
    print("GRAPH SMOTE SUMMARY")
    print("=" * 60)
    print(f"Original nodes: {G.number_of_nodes():,}")
    print(f"Balanced nodes: {G_balanced.number_of_nodes():,} (+{G_balanced.number_of_nodes() - G.number_of_nodes():,} synthetic)")
    print("\nOriginal phase distribution:")
    print(features_df['phase'].value_counts())
    print("\nBalanced phase distribution:")
    print(features_balanced['phase'].value_counts())
    print("=" * 60)


if __name__ == "__main__":
    main()
