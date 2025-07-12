"""
Graph Construction Utilities for Fraud Detection

This module handles:
- Converting tabular fraud data to graph structures
- Creating nodes from entities (cards, addresses, emails)
- Creating edges from transactions
- Building heterogeneous graphs with PyTorch Geometric
- Handling node and edge features
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import to_undirected
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
import logging
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudGraphBuilder:
    """
    Builds graph structures from fraud detection data
    """
    
    def __init__(self, node_types: List[str] = None):
        """
        Initialize the graph builder
        
        Args:
            node_types: List of node types to include in the graph
        """
        self.node_types = node_types or ['card', 'addr', 'email', 'device']
        self.node_mappings = {}
        self.node_features = {}
        self.edge_features = []
        self.edge_index = []
        self.edge_labels = []
        self.graph_stats = {}
        
    def extract_entities(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Extract unique entities from the dataset to create nodes
        
        Args:
            df: Preprocessed dataframe with transactions
            
        Returns:
            Dictionary mapping entity types to their unique values and features
        """
        logger.info("Extracting entities for graph nodes...")
        
        entities = {}
        
        # Card entities
        if 'card' in self.node_types:
            card_entities = {}
            
            # Primary card identifier
            if 'card1' in df.columns:
                unique_cards = df['card1'].dropna().unique()
                for card in unique_cards:
                    card_mask = df['card1'] == card
                    
                    # Aggregate features for this card
                    card_features = {
                        'transaction_count': card_mask.sum(),
                        'avg_amount': df.loc[card_mask, 'TransactionAmt'].mean() if 'TransactionAmt' in df.columns else 0,
                        'fraud_rate': df.loc[card_mask, 'isFraud'].mean() if 'isFraud' in df.columns else 0,
                        'unique_merchants': df.loc[card_mask, 'ProductCD'].nunique() if 'ProductCD' in df.columns else 0,
                    }
                    
                    # Add card type features if available
                    if 'card4' in df.columns:
                        card_type = df.loc[card_mask, 'card4'].mode().iloc[0] if len(df.loc[card_mask, 'card4'].mode()) > 0 else 'unknown'
                        card_features['card_type'] = card_type
                    
                    if 'card6' in df.columns:
                        card_category = df.loc[card_mask, 'card6'].mode().iloc[0] if len(df.loc[card_mask, 'card6'].mode()) > 0 else 'unknown'
                        card_features['card_category'] = card_category
                    
                    card_entities[f"card_{card}"] = card_features
            
            entities['card'] = card_entities
        
        # Address entities
        if 'addr' in self.node_types:
            addr_entities = {}
            
            for addr_col in ['addr1', 'addr2']:
                if addr_col in df.columns:
                    unique_addrs = df[addr_col].dropna().unique()
                    for addr in unique_addrs:
                        addr_mask = df[addr_col] == addr
                        
                        addr_features = {
                            'transaction_count': addr_mask.sum(),
                            'avg_amount': df.loc[addr_mask, 'TransactionAmt'].mean() if 'TransactionAmt' in df.columns else 0,
                            'fraud_rate': df.loc[addr_mask, 'isFraud'].mean() if 'isFraud' in df.columns else 0,
                            'unique_cards': df.loc[addr_mask, 'card1'].nunique() if 'card1' in df.columns else 0,
                        }
                        
                        addr_entities[f"{addr_col}_{addr}"] = addr_features
            
            entities['addr'] = addr_entities
        
        # Email domain entities
        if 'email' in self.node_types:
            email_entities = {}
            
            for email_col in ['P_emaildomain', 'R_emaildomain']:
                if email_col in df.columns:
                    unique_emails = df[email_col].dropna().unique()
                    for email in unique_emails:
                        email_mask = df[email_col] == email
                        
                        email_features = {
                            'transaction_count': email_mask.sum(),
                            'avg_amount': df.loc[email_mask, 'TransactionAmt'].mean() if 'TransactionAmt' in df.columns else 0,
                            'fraud_rate': df.loc[email_mask, 'isFraud'].mean() if 'isFraud' in df.columns else 0,
                            'unique_cards': df.loc[email_mask, 'card1'].nunique() if 'card1' in df.columns else 0,
                        }
                        
                        email_entities[f"{email_col}_{email}"] = email_features
            
            entities['email'] = email_entities
        
        # Device entities
        if 'device' in self.node_types:
            device_entities = {}
            
            for device_col in ['DeviceType', 'DeviceInfo']:
                if device_col in df.columns:
                    unique_devices = df[device_col].dropna().unique()
                    for device in unique_devices:
                        device_mask = df[device_col] == device
                        
                        device_features = {
                            'transaction_count': device_mask.sum(),
                            'avg_amount': df.loc[device_mask, 'TransactionAmt'].mean() if 'TransactionAmt' in df.columns else 0,
                            'fraud_rate': df.loc[device_mask, 'isFraud'].mean() if 'isFraud' in df.columns else 0,
                            'unique_cards': df.loc[device_mask, 'card1'].nunique() if 'card1' in df.columns else 0,
                        }
                        
                        device_entities[f"{device_col}_{device}"] = device_features
            
            entities['device'] = device_entities
        
        # Log entity statistics
        for entity_type, entity_dict in entities.items():
            logger.info(f"Extracted {len(entity_dict)} unique {entity_type} entities")
        
        return entities
    
    def create_node_mappings(self, entities: Dict[str, Dict]) -> Dict[str, Dict[str, int]]:
        """
        Create mappings from entity names to node indices
        
        Args:
            entities: Dictionary of entities by type
            
        Returns:
            Dictionary mapping entity names to indices
        """
        logger.info("Creating node index mappings...")
        
        mappings = {}
        
        for entity_type, entity_dict in entities.items():
            type_mapping = {}
            for idx, entity_name in enumerate(entity_dict.keys()):
                type_mapping[entity_name] = idx
            mappings[entity_type] = type_mapping
        
        self.node_mappings = mappings
        return mappings
    
    def create_node_features(self, entities: Dict[str, Dict]) -> Dict[str, torch.Tensor]:
        """
        Create node feature tensors from entity features
        
        Args:
            entities: Dictionary of entities by type
            
        Returns:
            Dictionary of node feature tensors by type
        """
        logger.info("Creating node feature tensors...")
        
        node_features = {}
        
        for entity_type, entity_dict in entities.items():
            if not entity_dict:
                continue
                
            # Get feature names from first entity
            feature_names = list(next(iter(entity_dict.values())).keys())
            n_entities = len(entity_dict)
            n_features = len(feature_names)
            
            # Create feature matrix
            features = torch.zeros(n_entities, n_features)
            
            for entity_name, entity_features in entity_dict.items():
                entity_idx = self.node_mappings[entity_type][entity_name]
                for feat_idx, feat_name in enumerate(feature_names):
                    # Handle categorical features
                    feat_value = entity_features[feat_name]
                    if isinstance(feat_value, str):
                        # Simple hash encoding for categorical features
                        feat_value = hash(feat_value) % 1000  # Limit range
                    features[entity_idx, feat_idx] = float(feat_value)
            
            node_features[entity_type] = features
            logger.info(f"Created {entity_type} node features: {features.shape}")
        
        self.node_features = node_features
        return node_features
    
    def create_edges_from_transactions(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create edges from transactions connecting involved entities
        
        Args:
            df: Dataframe with transaction data
            
        Returns:
            Tuple of (edge_index, edge_features, edge_labels)
        """
        logger.info("Creating edges from transactions...")
        
        edge_connections = []
        edge_features = []
        edge_labels = []
        
        # For each transaction, create edges between all involved entities
        for idx, row in df.iterrows():
            transaction_entities = []
            
            # Collect all entities involved in this transaction
            if 'card' in self.node_types and 'card1' in df.columns and pd.notna(row['card1']):
                entity_name = f"card_{row['card1']}"
                if entity_name in self.node_mappings.get('card', {}):
                    transaction_entities.append(('card', entity_name))
            
            # Add address entities
            if 'addr' in self.node_types:
                for addr_col in ['addr1', 'addr2']:
                    if addr_col in df.columns and pd.notna(row[addr_col]):
                        entity_name = f"{addr_col}_{row[addr_col]}"
                        if entity_name in self.node_mappings.get('addr', {}):
                            transaction_entities.append(('addr', entity_name))
            
            # Add email entities
            if 'email' in self.node_types:
                for email_col in ['P_emaildomain', 'R_emaildomain']:
                    if email_col in df.columns and pd.notna(row[email_col]):
                        entity_name = f"{email_col}_{row[email_col]}"
                        if entity_name in self.node_mappings.get('email', {}):
                            transaction_entities.append(('email', entity_name))
            
            # Add device entities
            if 'device' in self.node_types:
                for device_col in ['DeviceType', 'DeviceInfo']:
                    if device_col in df.columns and pd.notna(row[device_col]):
                        entity_name = f"{device_col}_{row[device_col]}"
                        if entity_name in self.node_mappings.get('device', {}):
                            transaction_entities.append(('device', entity_name))
            
            # Create edges between all pairs of entities in this transaction
            for i in range(len(transaction_entities)):
                for j in range(i + 1, len(transaction_entities)):
                    entity1_type, entity1_name = transaction_entities[i]
                    entity2_type, entity2_name = transaction_entities[j]
                    
                    entity1_idx = self.node_mappings[entity1_type][entity1_name]
                    entity2_idx = self.node_mappings[entity2_type][entity2_name]
                    
                    # Adjust indices for heterogeneous graph (offset by node type)
                    offset1 = sum(len(self.node_mappings[t]) for t in self.node_types[:self.node_types.index(entity1_type)])
                    offset2 = sum(len(self.node_mappings[t]) for t in self.node_types[:self.node_types.index(entity2_type)])
                    
                    global_idx1 = entity1_idx + offset1
                    global_idx2 = entity2_idx + offset2
                    
                    # Add both directions for undirected graph
                    edge_connections.extend([[global_idx1, global_idx2], [global_idx2, global_idx1]])
                    
                    # Transaction features for this edge
                    transaction_features = []
                    
                    # Amount features
                    if 'TransactionAmt' in df.columns:
                        transaction_features.append(float(row['TransactionAmt']) if pd.notna(row['TransactionAmt']) else 0.0)
                    
                    # Time features
                    if 'TransactionDT' in df.columns:
                        transaction_features.append(float(row['TransactionDT']) if pd.notna(row['TransactionDT']) else 0.0)
                    
                    # Product code
                    if 'ProductCD' in df.columns:
                        product_val = hash(str(row['ProductCD'])) % 100 if pd.notna(row['ProductCD']) else 0
                        transaction_features.append(float(product_val))
                    
                    # Add engineered features if available
                    for feature_col in ['TransactionHour', 'TransactionDayOfWeek', 'TransactionAmt_log']:
                        if feature_col in df.columns:
                            transaction_features.append(float(row[feature_col]) if pd.notna(row[feature_col]) else 0.0)
                    
                    # Ensure we have at least one feature
                    if not transaction_features:
                        transaction_features = [1.0]  # Default feature
                    
                    # Add features for both directions
                    edge_features.extend([transaction_features, transaction_features])
                    
                    # Edge labels (fraud or not)
                    edge_label = int(row['isFraud']) if 'isFraud' in df.columns and pd.notna(row['isFraud']) else 0
                    edge_labels.extend([edge_label, edge_label])
        
        if not edge_connections:
            logger.warning("No edges created from transactions!")
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1)), torch.empty(0, dtype=torch.long)
        
        # Convert to tensors
        edge_index = torch.tensor(edge_connections, dtype=torch.long).t().contiguous()
        edge_features_tensor = torch.tensor(edge_features, dtype=torch.float)
        edge_labels_tensor = torch.tensor(edge_labels, dtype=torch.long)
        
        logger.info(f"Created {edge_index.shape[1]} edges with {edge_features_tensor.shape[1]} features each")
        
        return edge_index, edge_features_tensor, edge_labels_tensor
    
    def build_homogeneous_graph(self, df: pd.DataFrame) -> Data:
        """
        Build a homogeneous graph where all nodes are treated equally
        
        Args:
            df: Preprocessed transaction dataframe
            
        Returns:
            PyTorch Geometric Data object
        """
        logger.info("Building homogeneous graph...")
        
        # Extract entities
        entities = self.extract_entities(df)
        
        # Create mappings
        self.create_node_mappings(entities)
        
        # Create node features
        node_features = self.create_node_features(entities)
        
        # Concatenate all node features (pad to same dimension for homogeneous graph)
        all_node_features = []
        max_features = 0
        
        # Find maximum feature dimension
        for entity_type in self.node_types:
            if entity_type in node_features:
                max_features = max(max_features, node_features[entity_type].shape[1])
        
        logger.info(f"Maximum feature dimension for homogeneous graph: {max_features}")
        
        # Pad all tensors to the same dimension
        for entity_type in self.node_types:
            if entity_type in node_features:
                features = node_features[entity_type]
                current_dim = features.shape[1]
                
                if current_dim < max_features:
                    # Pad with zeros to match max dimension
                    padding = torch.zeros(features.shape[0], max_features - current_dim)
                    features = torch.cat([features, padding], dim=1)
                    logger.info(f"Padded {entity_type} features from {current_dim} to {max_features} dimensions")
                
                all_node_features.append(features)
        
        if all_node_features:
            x = torch.cat(all_node_features, dim=0)
        else:
            x = torch.empty((0, 1))
        
        # Create edges
        edge_index, edge_attr, edge_labels = self.create_edges_from_transactions(df)
        
        # Create graph data
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=edge_labels,
            num_nodes=x.shape[0]
        )
        
        # Store graph statistics
        self.graph_stats = {
            'num_nodes': data.num_nodes,
            'num_edges': data.edge_index.shape[1],
            'num_node_features': data.x.shape[1],
            'num_edge_features': data.edge_attr.shape[1] if data.edge_attr is not None else 0,
            'fraud_rate': edge_labels.float().mean().item() if len(edge_labels) > 0 else 0.0
        }
        
        logger.info(f"Built homogeneous graph: {self.graph_stats}")
        
        return data
    
    def build_heterogeneous_graph(self, df: pd.DataFrame) -> HeteroData:
        """
        Build a heterogeneous graph with different node types
        
        Args:
            df: Preprocessed transaction dataframe
            
        Returns:
            PyTorch Geometric HeteroData object
        """
        logger.info("Building heterogeneous graph...")
        
        # Extract entities
        entities = self.extract_entities(df)
        
        # Create mappings
        self.create_node_mappings(entities)
        
        # Create node features
        node_features = self.create_node_features(entities)
        
        # Initialize heterogeneous data
        data = HeteroData()
        
        # Add node types and features
        for entity_type in self.node_types:
            if entity_type in node_features:
                data[entity_type].x = node_features[entity_type]
        
        # Create edges between different node types
        edge_types = []
        
        for idx, row in df.iterrows():
            # Get entities for this transaction
            transaction_entities = {}
            
            # Collect entities by type
            if 'card' in self.node_types and 'card1' in df.columns and pd.notna(row['card1']):
                entity_name = f"card_{row['card1']}"
                if entity_name in self.node_mappings.get('card', {}):
                    transaction_entities['card'] = self.node_mappings['card'][entity_name]
            
            if 'addr' in self.node_types:
                for addr_col in ['addr1', 'addr2']:
                    if addr_col in df.columns and pd.notna(row[addr_col]):
                        entity_name = f"{addr_col}_{row[addr_col]}"
                        if entity_name in self.node_mappings.get('addr', {}):
                            transaction_entities['addr'] = self.node_mappings['addr'][entity_name]
                            break  # Use first available address
            
            if 'email' in self.node_types:
                for email_col in ['P_emaildomain', 'R_emaildomain']:
                    if email_col in df.columns and pd.notna(row[email_col]):
                        entity_name = f"{email_col}_{row[email_col]}"
                        if entity_name in self.node_mappings.get('email', {}):
                            transaction_entities['email'] = self.node_mappings['email'][entity_name]
                            break  # Use first available email
            
            if 'device' in self.node_types:
                for device_col in ['DeviceType', 'DeviceInfo']:
                    if device_col in df.columns and pd.notna(row[device_col]):
                        entity_name = f"{device_col}_{row[device_col]}"
                        if entity_name in self.node_mappings.get('device', {}):
                            transaction_entities['device'] = self.node_mappings['device'][entity_name]
                            break  # Use first available device
            
            # Create edge features for this transaction
            edge_features = []
            if 'TransactionAmt' in df.columns:
                edge_features.append(float(row['TransactionAmt']) if pd.notna(row['TransactionAmt']) else 0.0)
            if 'TransactionDT' in df.columns:
                edge_features.append(float(row['TransactionDT']) if pd.notna(row['TransactionDT']) else 0.0)
            
            edge_label = int(row['isFraud']) if 'isFraud' in df.columns and pd.notna(row['isFraud']) else 0
            
            # Create edges between entity types
            entity_types_in_transaction = list(transaction_entities.keys())
            for i in range(len(entity_types_in_transaction)):
                for j in range(i + 1, len(entity_types_in_transaction)):
                    type1, type2 = entity_types_in_transaction[i], entity_types_in_transaction[j]
                    idx1, idx2 = transaction_entities[type1], transaction_entities[type2]
                    
                    edge_type = (type1, 'connects', type2)
                    reverse_edge_type = (type2, 'connects', type1)
                    
                    # Initialize edge tensors if they don't exist
                    if edge_type not in edge_types:
                        edge_types.append(edge_type)
                        data[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)
                        data[edge_type].edge_attr = torch.empty((0, len(edge_features)), dtype=torch.float)
                        data[edge_type].edge_label = torch.empty(0, dtype=torch.long)
                    
                    if reverse_edge_type not in edge_types:
                        edge_types.append(reverse_edge_type)
                        data[reverse_edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)
                        data[reverse_edge_type].edge_attr = torch.empty((0, len(edge_features)), dtype=torch.float)
                        data[reverse_edge_type].edge_label = torch.empty(0, dtype=torch.long)
                    
                    # Add edges
                    new_edge = torch.tensor([[idx1], [idx2]], dtype=torch.long)
                    new_reverse_edge = torch.tensor([[idx2], [idx1]], dtype=torch.long)
                    new_edge_attr = torch.tensor([edge_features], dtype=torch.float)
                    new_edge_label = torch.tensor([edge_label], dtype=torch.long)
                    
                    data[edge_type].edge_index = torch.cat([data[edge_type].edge_index, new_edge], dim=1)
                    data[edge_type].edge_attr = torch.cat([data[edge_type].edge_attr, new_edge_attr], dim=0)
                    data[edge_type].edge_label = torch.cat([data[edge_type].edge_label, new_edge_label], dim=0)
                    
                    data[reverse_edge_type].edge_index = torch.cat([data[reverse_edge_type].edge_index, new_reverse_edge], dim=1)
                    data[reverse_edge_type].edge_attr = torch.cat([data[reverse_edge_type].edge_attr, new_edge_attr], dim=0)
                    data[reverse_edge_type].edge_label = torch.cat([data[reverse_edge_type].edge_label, new_edge_label], dim=0)
        
        # Calculate statistics
        total_nodes = sum(data[node_type].x.shape[0] for node_type in self.node_types if node_type in data.node_types)
        total_edges = sum(data[edge_type].edge_index.shape[1] for edge_type in edge_types if edge_type in data.edge_types)
        
        self.graph_stats = {
            'num_nodes': total_nodes,
            'num_edges': total_edges,
            'num_node_types': len([nt for nt in self.node_types if nt in data.node_types]),
            'num_edge_types': len(edge_types),
            'node_types': list(data.node_types),
            'edge_types': list(data.edge_types)
        }
        
        logger.info(f"Built heterogeneous graph: {self.graph_stats}")
        
        return data
    
    def visualize_graph_statistics(self, save_path: str = None) -> None:
        """
        Create visualizations of graph statistics
        
        Args:
            save_path: Path to save the visualization
        """
        if not self.graph_stats:
            logger.warning("No graph statistics available for visualization")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Graph Statistics Overview', fontsize=16)
        
        # Basic statistics
        axes[0, 0].bar(['Nodes', 'Edges'], [self.graph_stats.get('num_nodes', 0), self.graph_stats.get('num_edges', 0)])
        axes[0, 0].set_title('Graph Size')
        axes[0, 0].set_ylabel('Count')
        
        # Node types (if heterogeneous)
        if 'node_types' in self.graph_stats:
            node_type_counts = []
            for node_type in self.graph_stats['node_types']:
                if node_type in self.node_features:
                    node_type_counts.append(self.node_features[node_type].shape[0])
            
            axes[0, 1].bar(self.graph_stats['node_types'], node_type_counts)
            axes[0, 1].set_title('Nodes by Type')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Fraud rate
        fraud_rate = self.graph_stats.get('fraud_rate', 0)
        axes[1, 0].pie([fraud_rate, 1 - fraud_rate], labels=['Fraud', 'Normal'], autopct='%1.1f%%')
        axes[1, 0].set_title('Fraud Distribution')
        
        # Feature dimensions
        feature_info = {
            'Node Features': self.graph_stats.get('num_node_features', 0),
            'Edge Features': self.graph_stats.get('num_edge_features', 0)
        }
        axes[1, 1].bar(feature_info.keys(), feature_info.values())
        axes[1, 1].set_title('Feature Dimensions')
        axes[1, 1].set_ylabel('Dimension')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Graph statistics saved to {save_path}")
        
        plt.show()


def create_graph_from_data(df: pd.DataFrame, 
                          graph_type: str = 'homogeneous',
                          node_types: List[str] = None) -> Union[Data, HeteroData]:
    """
    Convenience function to create graph from dataframe
    
    Args:
        df: Preprocessed transaction dataframe
        graph_type: Either 'homogeneous' or 'heterogeneous'
        node_types: List of node types to include
        
    Returns:
        PyTorch Geometric graph data
    """
    logger.info(f"Creating {graph_type} graph from transaction data...")
    
    builder = FraudGraphBuilder(node_types=node_types)
    
    if graph_type.lower() == 'homogeneous':
        return builder.build_homogeneous_graph(df)
    elif graph_type.lower() == 'heterogeneous':
        return builder.build_heterogeneous_graph(df)
    else:
        raise ValueError(f"Unknown graph type: {graph_type}. Must be 'homogeneous' or 'heterogeneous'")


if __name__ == "__main__":
    # Example usage
    from load_data import IEEE_CIS_DataLoader, create_sample_data
    import os
    
    # Create sample data if needed
    if not os.path.exists("../../data/train_transaction.csv"):
        create_sample_data("../../data/", n_samples=1000)
    
    # Load and preprocess data
    loader = IEEE_CIS_DataLoader("../../data/")
    data = loader.load_and_preprocess()
    
    # Create graphs
    homogeneous_graph = create_graph_from_data(data['full_data'], 'homogeneous')
    heterogeneous_graph = create_graph_from_data(data['full_data'], 'heterogeneous')
    
    print("Graph construction completed!")
    print(f"Homogeneous graph: {homogeneous_graph}")
    print(f"Heterogeneous graph: {heterogeneous_graph}") 