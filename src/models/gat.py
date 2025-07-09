"""
Graph Attention Network (GAT) Models for Fraud Detection

This module implements:
- Graph Attention Networks with multi-head attention
- Advanced attention mechanisms for fraud pattern learning
- Integration with the same training and evaluation pipeline
- Comparison capabilities with GCN models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm, global_mean_pool, global_max_pool
from torch_geometric.data import Data, HeteroData
from typing import Optional, Dict, List, Tuple, Union
import logging

# Import base components from GCN module
from .gcn import EdgePredictor, FocalLoss, WeightedFocalLoss

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GATLayer(nn.Module):
    """
    Single Graph Attention Layer with normalization and activation
    """
    
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 heads: int = 4,
                 concat: bool = True,
                 negative_slope: float = 0.2,
                 dropout: float = 0.0,
                 add_self_loops: bool = True,
                 bias: bool = True,
                 normalize: bool = True):
        """
        Initialize GAT layer
        
        Args:
            in_channels: Number of input features
            out_channels: Number of output features per head
            heads: Number of attention heads
            concat: Whether to concatenate or average attention heads
            negative_slope: LeakyReLU negative slope for attention
            dropout: Dropout rate
            add_self_loops: Whether to add self loops
            bias: Whether to use bias
            normalize: Whether to add batch normalization
        """
        super(GATLayer, self).__init__()
        
        self.heads = heads
        self.concat = concat
        
        self.gat = GATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=concat,
            negative_slope=negative_slope,
            dropout=dropout,
            add_self_loops=add_self_loops,
            bias=bias
        )
        
        # Output dimension depends on whether heads are concatenated
        final_out_channels = out_channels * heads if concat else out_channels
        self.batch_norm = BatchNorm(final_out_channels) if normalize else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
    def forward(self, 
                x: torch.Tensor, 
                edge_index: torch.Tensor,
                return_attention_weights: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Updated node features and optionally attention weights
        """
        # GAT convolution
        if return_attention_weights:
            x, attention_weights = self.gat(x, edge_index, return_attention_weights=True)
        else:
            x = self.gat(x, edge_index)
            attention_weights = None
        
        # Batch normalization
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        
        # Activation
        x = F.elu(x)  # ELU works well with GAT
        
        # Dropout
        if self.dropout is not None:
            x = self.dropout(x)
        
        if return_attention_weights:
            return x, attention_weights
        return x


class MultiLayerGAT(nn.Module):
    """
    Multi-layer Graph Attention Network
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int,
                 heads: List[int] = None,
                 dropout: float = 0.2,
                 batch_norm: bool = True,
                 attention_dropout: float = 0.1):
        """
        Initialize multi-layer GAT
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions (per head)
            output_dim: Output dimension
            heads: Number of attention heads per layer
            dropout: Dropout rate
            batch_norm: Whether to use batch normalization
            attention_dropout: Dropout rate for attention weights
        """
        super(MultiLayerGAT, self).__init__()
        
        self.num_layers = len(hidden_dims) + 1
        self.layers = nn.ModuleList()
        
        # Default attention heads if not specified
        if heads is None:
            heads = [4] * len(hidden_dims) + [1]  # Single head for output layer
        
        # Input layer
        if hidden_dims:
            self.layers.append(
                GATLayer(
                    in_channels=input_dim,
                    out_channels=hidden_dims[0],
                    heads=heads[0],
                    concat=True,  # Concatenate for hidden layers
                    dropout=attention_dropout,
                    normalize=batch_norm
                )
            )
            
            current_dim = hidden_dims[0] * heads[0]
            
            # Hidden layers
            for i in range(1, len(hidden_dims)):
                self.layers.append(
                    GATLayer(
                        in_channels=current_dim,
                        out_channels=hidden_dims[i],
                        heads=heads[i],
                        concat=True,
                        dropout=attention_dropout,
                        normalize=batch_norm
                    )
                )
                current_dim = hidden_dims[i] * heads[i]
            
            # Output layer (average attention heads)
            self.layers.append(
                GATLayer(
                    in_channels=current_dim,
                    out_channels=output_dim,
                    heads=heads[-1],
                    concat=False,  # Average for output layer
                    dropout=0.0,
                    normalize=False
                )
            )
        else:
            # Single layer
            self.layers.append(
                GATLayer(
                    in_channels=input_dim,
                    out_channels=output_dim,
                    heads=heads[0],
                    concat=False,
                    dropout=0.0,
                    normalize=False
                )
            )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                x: torch.Tensor, 
                edge_index: torch.Tensor,
                return_attention_weights: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass through all layers
        
        Args:
            x: Node features
            edge_index: Edge indices
            return_attention_weights: Whether to return attention weights from all layers
            
        Returns:
            Node embeddings and optionally attention weights
        """
        attention_weights_all = []
        
        for i, layer in enumerate(self.layers):
            if return_attention_weights and i < len(self.layers) - 1:  # Don't return attention for output layer
                x, attention_weights = layer(x, edge_index, return_attention_weights=True)
                attention_weights_all.append(attention_weights)
            else:
                x = layer(x, edge_index)
            
            # Add residual connections for deeper networks
            if i > 0 and x.shape[-1] == self.layers[i-1].gat.out_channels * self.layers[i-1].heads:
                # Residual connection if dimensions match
                pass  # Could implement if needed
        
        if return_attention_weights:
            return x, attention_weights_all
        return x


class FraudGAT(nn.Module):
    """
    Complete GAT model for fraud detection
    """
    
    def __init__(self,
                 node_feature_dim: int,
                 edge_feature_dim: int = 0,
                 hidden_dims: List[int] = [64, 32],
                 heads: List[int] = None,
                 embedding_dim: int = 32,
                 num_classes: int = 2,
                 dropout: float = 0.2,
                 attention_dropout: float = 0.1,
                 batch_norm: bool = True,
                 task: str = 'edge_classification'):
        """
        Initialize Fraud GAT model
        
        Args:
            node_feature_dim: Dimension of node features
            edge_feature_dim: Dimension of edge features
            hidden_dims: Hidden layer dimensions for GAT
            heads: Number of attention heads per layer
            embedding_dim: Final node embedding dimension
            num_classes: Number of output classes
            dropout: Dropout rate
            attention_dropout: Attention dropout rate
            batch_norm: Whether to use batch normalization
            task: Either 'node_classification' or 'edge_classification'
        """
        super(FraudGAT, self).__init__()
        
        self.task = task
        self.embedding_dim = embedding_dim
        
        # Default attention heads
        if heads is None:
            heads = [8, 4, 1]  # Start with more heads, reduce in deeper layers
        
        # GAT backbone
        self.gat = MultiLayerGAT(
            input_dim=node_feature_dim,
            hidden_dims=hidden_dims,
            output_dim=embedding_dim,
            heads=heads,
            dropout=dropout,
            batch_norm=batch_norm,
            attention_dropout=attention_dropout
        )
        
        if task == 'edge_classification':
            # Edge classification head (reuse from GCN)
            self.edge_predictor = EdgePredictor(
                node_embedding_dim=embedding_dim,
                edge_feature_dim=edge_feature_dim,
                hidden_dim=64,
                num_classes=num_classes,
                dropout=dropout
            )
        elif task == 'node_classification':
            # Node classification head
            self.node_classifier = nn.Sequential(
                nn.Linear(embedding_dim, 64),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_classes)
            )
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def forward(self, 
                x: torch.Tensor, 
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None,
                return_attention_weights: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, node_feature_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_feature_dim]
            batch: Batch indices for graph-level tasks
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Predictions and optionally attention weights
        """
        # Get node embeddings through GAT
        if return_attention_weights:
            node_embeddings, attention_weights = self.gat(x, edge_index, return_attention_weights=True)
        else:
            node_embeddings = self.gat(x, edge_index)
            attention_weights = None
        
        if self.task == 'edge_classification':
            # Edge-level predictions
            predictions = self.edge_predictor(node_embeddings, edge_index, edge_attr)
        elif self.task == 'node_classification':
            # Node-level predictions
            predictions = self.node_classifier(node_embeddings)
        else:
            # Return embeddings for other downstream tasks
            predictions = node_embeddings
        
        if return_attention_weights:
            return predictions, attention_weights
        return predictions


class AttentionVisualizer:
    """
    Utility class for visualizing attention weights
    """
    
    def __init__(self, model: FraudGAT):
        self.model = model
    
    def get_attention_weights(self, data) -> List[torch.Tensor]:
        """
        Extract attention weights from model
        
        Args:
            data: Graph data
            
        Returns:
            List of attention weights from each layer
        """
        self.model.eval()
        with torch.no_grad():
            _, attention_weights = self.model(
                data.x, data.edge_index, data.edge_attr, 
                return_attention_weights=True
            )
        return attention_weights
    
    def visualize_attention_distribution(self, attention_weights: List[torch.Tensor], layer_idx: int = 0):
        """
        Visualize attention weight distribution
        
        Args:
            attention_weights: Attention weights from model
            layer_idx: Which layer to visualize
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        if layer_idx >= len(attention_weights):
            logger.warning(f"Layer {layer_idx} not available. Using layer 0.")
            layer_idx = 0
        
        weights = attention_weights[layer_idx].cpu().numpy()
        
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.hist(weights.flatten(), bins=50, alpha=0.7)
        plt.title(f'Attention Weight Distribution (Layer {layer_idx})')
        plt.xlabel('Attention Weight')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 2, 2)
        plt.boxplot([weights[:, i] for i in range(min(8, weights.shape[1]))], 
                   labels=[f'Head {i}' for i in range(min(8, weights.shape[1]))])
        plt.title(f'Attention Weights by Head (Layer {layer_idx})')
        plt.ylabel('Attention Weight')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()


def create_fraud_gat(node_feature_dim: int,
                    edge_feature_dim: int = 0,
                    model_config: Optional[Dict] = None) -> FraudGAT:
    """
    Factory function to create FraudGAT model with default configuration
    
    Args:
        node_feature_dim: Dimension of node features
        edge_feature_dim: Dimension of edge features
        model_config: Model configuration dictionary
        
    Returns:
        Configured FraudGAT model
    """
    default_config = {
        'hidden_dims': [64, 32],
        'heads': [8, 4, 1],
        'embedding_dim': 32,
        'num_classes': 2,
        'dropout': 0.2,
        'attention_dropout': 0.1,
        'batch_norm': True,
        'task': 'edge_classification'
    }
    
    if model_config:
        default_config.update(model_config)
    
    model = FraudGAT(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        **default_config
    )
    
    logger.info(f"Created FraudGAT model with config: {default_config}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def compare_gat_gcn_performance(gat_model: FraudGAT, 
                               gcn_model,
                               test_data,
                               device: str = 'cpu') -> Dict[str, Dict[str, float]]:
    """
    Compare GAT and GCN model performance
    
    Args:
        gat_model: Trained GAT model
        gcn_model: Trained GCN model
        test_data: Test data
        device: Device to run on
        
    Returns:
        Comparison results
    """
    from ..evaluate import FraudDetectionEvaluator
    
    results = {}
    
    # Evaluate GAT
    gat_evaluator = FraudDetectionEvaluator(gat_model, device)
    gat_results = gat_evaluator.evaluate_model(test_data)
    results['GAT'] = gat_results.metrics
    
    # Evaluate GCN
    gcn_evaluator = FraudDetectionEvaluator(gcn_model, device)
    gcn_results = gcn_evaluator.evaluate_model(test_data)
    results['GCN'] = gcn_results.metrics
    
    # Print comparison
    print("🔍 GAT vs GCN Performance Comparison")
    print("=" * 50)
    
    key_metrics = ['roc_auc', 'pr_auc', 'f1', 'precision', 'recall']
    for metric in key_metrics:
        gat_score = results['GAT'].get(metric, 0)
        gcn_score = results['GCN'].get(metric, 0)
        winner = "GAT" if gat_score > gcn_score else "GCN"
        
        print(f"{metric.upper():12s}: GAT={gat_score:.4f}, GCN={gcn_score:.4f} → {winner} wins")
    
    return results


if __name__ == "__main__":
    # Example usage and testing
    torch.manual_seed(42)
    
    # Create sample data
    num_nodes = 1000
    num_edges = 5000
    node_feature_dim = 16
    edge_feature_dim = 8
    
    # Sample graph data
    x = torch.randn(num_nodes, node_feature_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, edge_feature_dim)
    edge_labels = torch.randint(0, 2, (num_edges,))  # Binary fraud labels
    
    # Create GAT model
    model = create_fraud_gat(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        model_config={
            'hidden_dims': [64, 32], 
            'heads': [8, 4, 1],
            'dropout': 0.1
        }
    )
    
    # Forward pass
    predictions = model(x, edge_index, edge_attr)
    predictions_with_attention, attention_weights = model(
        x, edge_index, edge_attr, return_attention_weights=True
    )
    
    print(f"Input shape: {x.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Edge features shape: {edge_attr.shape}")
    print(f"Output shape: {predictions.shape}")
    print(f"Number of attention layers: {len(attention_weights)}")
    
    # Test attention visualization
    visualizer = AttentionVisualizer(model)
    attention_weights = visualizer.get_attention_weights(
        type('Data', (), {'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr})()
    )
    
    print("GAT model test completed successfully!") 