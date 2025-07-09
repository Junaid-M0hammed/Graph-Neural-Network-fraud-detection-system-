"""
Graph Convolutional Network (GCN) Models for Fraud Detection

This module implements:
- Basic GCN architecture for node and edge classification
- Multi-layer GCN with proper regularization
- Handling of class imbalance in fraud detection
- Support for homogeneous and heterogeneous graphs
- Edge-level prediction for transaction fraud detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm, global_mean_pool, global_max_pool
from torch_geometric.data import Data, HeteroData, Batch
from torch_geometric.utils import to_edge_index, add_self_loops
from typing import Optional, Dict, List, Tuple, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GCNLayer(nn.Module):
    """
    Single Graph Convolutional Layer with normalization and activation
    """
    
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 improved: bool = False,
                 cached: bool = False,
                 add_self_loops: bool = True,
                 normalize: bool = True,
                 bias: bool = True,
                 dropout: float = 0.0):
        """
        Initialize GCN layer
        
        Args:
            in_channels: Number of input features
            out_channels: Number of output features
            improved: Whether to use improved GCN formulation
            cached: Whether to cache the computation
            add_self_loops: Whether to add self loops
            normalize: Whether to add normalization
            bias: Whether to use bias
            dropout: Dropout rate
        """
        super(GCNLayer, self).__init__()
        
        self.gcn = GCNConv(
            in_channels=in_channels,
            out_channels=out_channels,
            improved=improved,
            cached=cached,
            add_self_loops=add_self_loops,
            normalize=True,
            bias=bias
        )
        
        self.batch_norm = BatchNorm(out_channels) if normalize else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Updated node features [num_nodes, out_channels]
        """
        # GCN convolution
        x = self.gcn(x, edge_index)
        
        # Batch normalization
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        
        # Activation
        x = F.relu(x)
        
        # Dropout
        if self.dropout is not None:
            x = self.dropout(x)
        
        return x


class MultiLayerGCN(nn.Module):
    """
    Multi-layer Graph Convolutional Network
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int,
                 dropout: float = 0.2,
                 batch_norm: bool = True):
        """
        Initialize multi-layer GCN
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            dropout: Dropout rate
            batch_norm: Whether to use batch normalization
        """
        super(MultiLayerGCN, self).__init__()
        
        self.num_layers = len(hidden_dims) + 1
        self.layers = nn.ModuleList()
        
        # Input layer
        if hidden_dims:
            self.layers.append(
                GCNLayer(
                    in_channels=input_dim,
                    out_channels=hidden_dims[0],
                    normalize=batch_norm,
                    dropout=dropout
                )
            )
            
            # Hidden layers
            for i in range(1, len(hidden_dims)):
                self.layers.append(
                    GCNLayer(
                        in_channels=hidden_dims[i-1],
                        out_channels=hidden_dims[i],
                        normalize=batch_norm,
                        dropout=dropout
                    )
                )
            
            # Output layer (no activation, no dropout)
            self.layers.append(
                GCNLayer(
                    in_channels=hidden_dims[-1],
                    out_channels=output_dim,
                    normalize=False,
                    dropout=0.0
                )
            )
        else:
            # Single layer
            self.layers.append(
                GCNLayer(
                    in_channels=input_dim,
                    out_channels=output_dim,
                    normalize=False,
                    dropout=0.0
                )
            )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all layers
        
        Args:
            x: Node features
            edge_index: Edge indices
            
        Returns:
            Node embeddings
        """
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            
            # No activation on final layer
            if i == len(self.layers) - 1:
                x = self.layers[-1].gcn(x, edge_index)  # Raw output without activation
        
        return x


class EdgePredictor(nn.Module):
    """
    Edge prediction head for fraud detection
    """
    
    def __init__(self,
                 node_embedding_dim: int,
                 edge_feature_dim: int = 0,
                 hidden_dim: int = 64,
                 num_classes: int = 2,
                 dropout: float = 0.2):
        """
        Initialize edge predictor
        
        Args:
            node_embedding_dim: Dimension of node embeddings
            edge_feature_dim: Dimension of edge features
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes (2 for binary fraud detection)
            dropout: Dropout rate
        """
        super(EdgePredictor, self).__init__()
        
        # Combine node embeddings (2 nodes per edge) with edge features
        input_dim = 2 * node_embedding_dim + edge_feature_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, 
                node_embeddings: torch.Tensor, 
                edge_index: torch.Tensor,
                edge_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict edge labels
        
        Args:
            node_embeddings: Node embeddings [num_nodes, embedding_dim]
            edge_index: Edge indices [2, num_edges]
            edge_features: Edge features [num_edges, edge_feature_dim]
            
        Returns:
            Edge predictions [num_edges, num_classes]
        """
        # Get embeddings for source and target nodes
        src_embeddings = node_embeddings[edge_index[0]]  # [num_edges, embedding_dim]
        dst_embeddings = node_embeddings[edge_index[1]]  # [num_edges, embedding_dim]
        
        # Concatenate source and destination embeddings
        edge_representations = torch.cat([src_embeddings, dst_embeddings], dim=1)
        
        # Add edge features if available
        if edge_features is not None:
            edge_representations = torch.cat([edge_representations, edge_features], dim=1)
        
        # Predict edge labels
        return self.mlp(edge_representations)


class FraudGCN(nn.Module):
    """
    Complete GCN model for fraud detection
    """
    
    def __init__(self,
                 node_feature_dim: int,
                 edge_feature_dim: int = 0,
                 hidden_dims: List[int] = [64, 32],
                 embedding_dim: int = 32,
                 num_classes: int = 2,
                 dropout: float = 0.2,
                 batch_norm: bool = True,
                 task: str = 'edge_classification'):
        """
        Initialize Fraud GCN model
        
        Args:
            node_feature_dim: Dimension of node features
            edge_feature_dim: Dimension of edge features
            hidden_dims: Hidden layer dimensions for GCN
            embedding_dim: Final node embedding dimension
            num_classes: Number of output classes
            dropout: Dropout rate
            batch_norm: Whether to use batch normalization
            task: Either 'node_classification' or 'edge_classification'
        """
        super(FraudGCN, self).__init__()
        
        self.task = task
        self.embedding_dim = embedding_dim
        
        # GCN backbone
        self.gcn = MultiLayerGCN(
            input_dim=node_feature_dim,
            hidden_dims=hidden_dims,
            output_dim=embedding_dim,
            dropout=dropout,
            batch_norm=batch_norm
        )
        
        if task == 'edge_classification':
            # Edge classification head
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
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_classes)
            )
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def forward(self, 
                x: torch.Tensor, 
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, node_feature_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_feature_dim]
            batch: Batch indices for graph-level tasks
            
        Returns:
            Predictions based on task type
        """
        # Get node embeddings through GCN
        node_embeddings = self.gcn(x, edge_index)
        
        if self.task == 'edge_classification':
            # Edge-level predictions
            return self.edge_predictor(node_embeddings, edge_index, edge_attr)
        
        elif self.task == 'node_classification':
            # Node-level predictions
            return self.node_classifier(node_embeddings)
        
        else:
            # Return embeddings for other downstream tasks
            return node_embeddings


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in fraud detection
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss
        
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
            reduction: Reduction method
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Loss
        
        Args:
            inputs: Model predictions [batch_size, num_classes]
            targets: True labels [batch_size]
            
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedFocalLoss(nn.Module):
    """
    Weighted Focal Loss for extreme class imbalance
    """
    
    def __init__(self, 
                 class_weights: Optional[torch.Tensor] = None,
                 alpha: float = 1.0, 
                 gamma: float = 2.0, 
                 reduction: str = 'mean'):
        """
        Initialize Weighted Focal Loss
        
        Args:
            class_weights: Weights for each class
            alpha: Focusing parameter alpha
            gamma: Focusing parameter gamma
            reduction: Reduction method
        """
        super(WeightedFocalLoss, self).__init__()
        self.class_weights = class_weights
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Weighted Focal Loss
        """
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def create_fraud_gcn(node_feature_dim: int,
                    edge_feature_dim: int = 0,
                    model_config: Optional[Dict] = None) -> FraudGCN:
    """
    Factory function to create FraudGCN model with default configuration
    
    Args:
        node_feature_dim: Dimension of node features
        edge_feature_dim: Dimension of edge features
        model_config: Model configuration dictionary
        
    Returns:
        Configured FraudGCN model
    """
    default_config = {
        'hidden_dims': [128, 64, 32],
        'embedding_dim': 32,
        'num_classes': 2,
        'dropout': 0.2,
        'batch_norm': True,
        'task': 'edge_classification'
    }
    
    if model_config:
        default_config.update(model_config)
    
    model = FraudGCN(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        **default_config
    )
    
    logger.info(f"Created FraudGCN model with config: {default_config}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def compute_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets
    
    Args:
        labels: Class labels
        
    Returns:
        Class weights tensor
    """
    unique_labels, counts = torch.unique(labels, return_counts=True)
    total_samples = len(labels)
    
    # Inverse frequency weighting
    weights = total_samples / (len(unique_labels) * counts.float())
    
    logger.info(f"Class distribution: {dict(zip(unique_labels.tolist(), counts.tolist()))}")
    logger.info(f"Computed class weights: {dict(zip(unique_labels.tolist(), weights.tolist()))}")
    
    return weights


def get_loss_function(loss_type: str = 'focal',
                     class_weights: Optional[torch.Tensor] = None,
                     **kwargs) -> nn.Module:
    """
    Get appropriate loss function for fraud detection
    
    Args:
        loss_type: Type of loss ('ce', 'focal', 'weighted_focal')
        class_weights: Class weights for imbalanced data
        **kwargs: Additional arguments for loss function
        
    Returns:
        Loss function
    """
    if loss_type == 'ce':
        return nn.CrossEntropyLoss(weight=class_weights)
    
    elif loss_type == 'focal':
        return FocalLoss(
            alpha=kwargs.get('alpha', 1.0),
            gamma=kwargs.get('gamma', 2.0)
        )
    
    elif loss_type == 'weighted_focal':
        return WeightedFocalLoss(
            class_weights=class_weights,
            alpha=kwargs.get('alpha', 1.0),
            gamma=kwargs.get('gamma', 2.0)
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


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
    
    # Create model
    model = create_fraud_gcn(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        model_config={'hidden_dims': [64, 32], 'dropout': 0.1}
    )
    
    # Forward pass
    predictions = model(x, edge_index, edge_attr)
    
    print(f"Input shape: {x.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Edge features shape: {edge_attr.shape}")
    print(f"Output shape: {predictions.shape}")
    
    # Test loss function
    class_weights = compute_class_weights(edge_labels)
    loss_fn = get_loss_function('weighted_focal', class_weights=class_weights)
    loss = loss_fn(predictions, edge_labels)
    
    print(f"Loss: {loss.item():.4f}")
    print("Model test completed successfully!") 