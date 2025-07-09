"""
Training Pipeline for GCN Fraud Detection Model

This module implements:
- Complete training and validation loops
- Model checkpointing and early stopping
- Learning rate scheduling
- Comprehensive metric tracking
- Support for different graph types and data loaders
- Experiment logging and visualization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.loader import NeighborLoader
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import logging
import os
import time
import json
from collections import defaultdict
import warnings

# Import our modules
from models.gcn import FraudGCN, create_fraud_gcn, get_loss_function, compute_class_weights
from utils.load_data import IEEE_CIS_DataLoader
from utils.build_graph import FraudGraphBuilder, create_graph_from_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class MetricTracker:
    """
    Tracks training and validation metrics
    """
    
    def __init__(self):
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        self.best_metrics = {}
        
    def update_train(self, metrics: Dict[str, float]):
        """Update training metrics"""
        for key, value in metrics.items():
            self.train_metrics[key].append(value)
    
    def update_val(self, metrics: Dict[str, float]):
        """Update validation metrics"""
        for key, value in metrics.items():
            self.val_metrics[key].append(value)
    
    def get_latest(self, split: str = 'val') -> Dict[str, float]:
        """Get latest metrics"""
        metrics = self.val_metrics if split == 'val' else self.train_metrics
        return {key: values[-1] for key, values in metrics.items() if values}
    
    def is_best(self, metric: str, higher_better: bool = True) -> bool:
        """Check if current metric is the best so far"""
        if metric not in self.val_metrics or not self.val_metrics[metric]:
            return False
        
        current_value = self.val_metrics[metric][-1]
        
        if metric not in self.best_metrics:
            self.best_metrics[metric] = current_value
            return True
        
        if higher_better:
            is_better = current_value > self.best_metrics[metric]
        else:
            is_better = current_value < self.best_metrics[metric]
        
        if is_better:
            self.best_metrics[metric] = current_value
        
        return is_better


class EarlyStopping:
    """
    Early stopping to prevent overfitting
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, higher_better: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.higher_better = higher_better
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_score: float) -> bool:
        if self.best_score is None:
            self.best_score = val_score
        elif self._is_improvement(val_score):
            self.best_score = val_score
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            
        return self.early_stop
    
    def _is_improvement(self, score: float) -> bool:
        if self.higher_better:
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta


class FraudGCNTrainer:
    """
    Complete training pipeline for fraud detection GCN
    """
    
    def __init__(self,
                 model: FraudGCN,
                 device: str = 'cpu',
                 checkpoint_dir: str = 'checkpoints/',
                 experiment_name: str = 'fraud_gcn_experiment'):
        """
        Initialize trainer
        
        Args:
            model: GCN model to train
            device: Device to use for training
            checkpoint_dir: Directory to save checkpoints
            experiment_name: Name of the experiment
        """
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.experiment_name = experiment_name
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize components
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.metric_tracker = MetricTracker()
        self.early_stopping = None
        
        # Training state
        self.epoch = 0
        self.best_model_path = None
        
    def setup_training(self,
                      optimizer_config: Dict = None,
                      scheduler_config: Dict = None,
                      loss_config: Dict = None,
                      early_stopping_config: Dict = None):
        """
        Setup training components
        
        Args:
            optimizer_config: Optimizer configuration
            scheduler_config: Scheduler configuration
            loss_config: Loss function configuration
            early_stopping_config: Early stopping configuration
        """
        # Default configurations
        optimizer_config = optimizer_config or {'type': 'adam', 'lr': 0.001, 'weight_decay': 1e-5}
        scheduler_config = scheduler_config or {'type': 'reduce_on_plateau', 'patience': 5, 'factor': 0.5}
        loss_config = loss_config or {'type': 'focal', 'alpha': 1.0, 'gamma': 2.0}
        early_stopping_config = early_stopping_config or {'patience': 15, 'min_delta': 0.001}
        
        # Setup optimizer
        if optimizer_config['type'].lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config.get('weight_decay', 1e-5)
            )
        elif optimizer_config['type'].lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config.get('weight_decay', 1e-5)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_config['type']}")
        
        # Setup scheduler
        if scheduler_config['type'].lower() == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                patience=scheduler_config.get('patience', 5),
                factor=scheduler_config.get('factor', 0.5),
                verbose=True
            )
        elif scheduler_config['type'].lower() == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.get('T_max', 100),
                eta_min=scheduler_config.get('eta_min', 1e-6)
            )
        
        # Setup loss function
        self.loss_fn = get_loss_function(**loss_config)
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(**early_stopping_config)
        
        logger.info("Training setup completed")
        logger.info(f"Optimizer: {optimizer_config}")
        logger.info(f"Scheduler: {scheduler_config}")
        logger.info(f"Loss: {loss_config}")
    
    def compute_metrics(self, 
                       predictions: torch.Tensor, 
                       targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics
        
        Args:
            predictions: Model predictions [batch_size, num_classes]
            targets: True labels [batch_size]
            
        Returns:
            Dictionary of metrics
        """
        # Convert to numpy
        if predictions.dim() > 1:
            probs = torch.softmax(predictions, dim=1)[:, 1].detach().cpu().numpy()
            preds = torch.argmax(predictions, dim=1).detach().cpu().numpy()
        else:
            probs = torch.sigmoid(predictions).detach().cpu().numpy()
            preds = (probs > 0.5).astype(int)
        
        targets_np = targets.detach().cpu().numpy()
        
        # Basic metrics
        accuracy = accuracy_score(targets_np, preds)
        precision = precision_score(targets_np, preds, zero_division=0)
        recall = recall_score(targets_np, preds, zero_division=0)
        f1 = f1_score(targets_np, preds, zero_division=0)
        
        # ROC AUC
        try:
            roc_auc = roc_auc_score(targets_np, probs)
        except ValueError:
            roc_auc = 0.0
        
        # PR AUC
        try:
            precision_curve, recall_curve, _ = precision_recall_curve(targets_np, probs)
            pr_auc = auc(recall_curve, precision_curve)
        except ValueError:
            pr_auc = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }
    
    def train_epoch(self, data: Data) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            data: Graph data
            
        Returns:
            Training metrics
        """
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        # For simplicity, we'll train on the entire graph
        # In practice, you might want to use mini-batching for large graphs
        
        self.optimizer.zero_grad()
        
        # Forward pass
        predictions = self.model(data.x, data.edge_index, data.edge_attr)
        
        # Compute loss
        loss = self.loss_fn(predictions, data.y)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Collect predictions and targets
        all_predictions.append(predictions)
        all_targets.append(data.y)
        
        # Compute metrics
        epoch_predictions = torch.cat(all_predictions, dim=0)
        epoch_targets = torch.cat(all_targets, dim=0)
        
        metrics = self.compute_metrics(epoch_predictions, epoch_targets)
        metrics['loss'] = loss.item()
        
        return metrics
    
    def validate_epoch(self, data: Data) -> Dict[str, float]:
        """
        Validate for one epoch
        
        Args:
            data: Graph data
            
        Returns:
            Validation metrics
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        total_loss = 0
        
        with torch.no_grad():
            # Forward pass
            predictions = self.model(data.x, data.edge_index, data.edge_attr)
            
            # Compute loss
            loss = self.loss_fn(predictions, data.y)
            total_loss += loss.item()
            
            # Collect predictions and targets
            all_predictions.append(predictions)
            all_targets.append(data.y)
        
        # Compute metrics
        epoch_predictions = torch.cat(all_predictions, dim=0)
        epoch_targets = torch.cat(all_targets, dim=0)
        
        metrics = self.compute_metrics(epoch_predictions, epoch_targets)
        metrics['loss'] = total_loss
        
        return metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """
        Save model checkpoint
        
        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metric_tracker': self.metric_tracker,
            'best_metrics': self.metric_tracker.best_metrics
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f'{self.experiment_name}_epoch_{self.epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(
                self.checkpoint_dir,
                f'{self.experiment_name}_best.pth'
            )
            torch.save(checkpoint, best_path)
            self.best_model_path = best_path
            logger.info(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.metric_tracker = checkpoint['metric_tracker']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def train(self,
              train_data: Data,
              val_data: Data,
              num_epochs: int = 100,
              save_every: int = 10,
              log_every: int = 1) -> Dict[str, List[float]]:
        """
        Complete training loop
        
        Args:
            train_data: Training graph data
            val_data: Validation graph data
            num_epochs: Number of training epochs
            save_every: Save checkpoint every N epochs
            log_every: Log metrics every N epochs
            
        Returns:
            Training history
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.epoch = epoch + 1
            
            # Training
            train_metrics = self.train_epoch(train_data)
            self.metric_tracker.update_train(train_metrics)
            
            # Validation
            val_metrics = self.validate_epoch(val_data)
            self.metric_tracker.update_val(val_metrics)
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['roc_auc'])
                else:
                    self.scheduler.step()
            
            # Check if this is the best model
            is_best = self.metric_tracker.is_best('roc_auc', higher_better=True)
            
            # Logging
            if epoch % log_every == 0:
                logger.info(
                    f"Epoch {self.epoch:3d}/{num_epochs} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Train AUC: {train_metrics['roc_auc']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val AUC: {val_metrics['roc_auc']:.4f} | "
                    f"Val F1: {val_metrics['f1']:.4f} | "
                    f"Best: {is_best}"
                )
            
            # Save checkpoints
            if epoch % save_every == 0 or is_best:
                self.save_checkpoint(is_best=is_best)
            
            # Early stopping
            if self.early_stopping(val_metrics['roc_auc']):
                logger.info(f"Early stopping triggered at epoch {self.epoch}")
                break
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        logger.info(f"Best validation metrics: {self.metric_tracker.best_metrics}")
        
        return {
            'train_metrics': dict(self.metric_tracker.train_metrics),
            'val_metrics': dict(self.metric_tracker.val_metrics),
            'best_metrics': self.metric_tracker.best_metrics
        }
    
    def plot_training_history(self, save_path: str = None):
        """
        Plot training history
        
        Args:
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training History - {self.experiment_name}', fontsize=16)
        
        # Loss
        epochs = range(1, len(self.metric_tracker.train_metrics['loss']) + 1)
        axes[0, 0].plot(epochs, self.metric_tracker.train_metrics['loss'], label='Train')
        axes[0, 0].plot(epochs, self.metric_tracker.val_metrics['loss'], label='Validation')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # ROC AUC
        axes[0, 1].plot(epochs, self.metric_tracker.train_metrics['roc_auc'], label='Train')
        axes[0, 1].plot(epochs, self.metric_tracker.val_metrics['roc_auc'], label='Validation')
        axes[0, 1].set_title('ROC AUC')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 Score
        axes[1, 0].plot(epochs, self.metric_tracker.train_metrics['f1'], label='Train')
        axes[1, 0].plot(epochs, self.metric_tracker.val_metrics['f1'], label='Validation')
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Precision-Recall
        axes[1, 1].plot(epochs, self.metric_tracker.train_metrics['precision'], label='Train Precision')
        axes[1, 1].plot(epochs, self.metric_tracker.val_metrics['precision'], label='Val Precision')
        axes[1, 1].plot(epochs, self.metric_tracker.train_metrics['recall'], label='Train Recall')
        axes[1, 1].plot(epochs, self.metric_tracker.val_metrics['recall'], label='Val Recall')
        axes[1, 1].set_title('Precision & Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()


def create_train_val_split(data: Data, 
                          val_ratio: float = 0.2, 
                          random_state: int = 42) -> Tuple[Data, Data]:
    """
    Split graph data into training and validation sets
    
    Args:
        data: Graph data
        val_ratio: Validation set ratio
        random_state: Random seed
        
    Returns:
        Tuple of (train_data, val_data)
    """
    torch.manual_seed(random_state)
    
    # Create edge masks for train/val split
    num_edges = data.edge_index.shape[1]
    perm = torch.randperm(num_edges)
    
    num_val = int(val_ratio * num_edges)
    val_mask = perm[:num_val]
    train_mask = perm[num_val:]
    
    # Create training data
    train_data = Data(
        x=data.x,
        edge_index=data.edge_index[:, train_mask],
        edge_attr=data.edge_attr[train_mask] if data.edge_attr is not None else None,
        y=data.y[train_mask] if data.y is not None else None,
        num_nodes=data.num_nodes
    )
    
    # Create validation data
    val_data = Data(
        x=data.x,
        edge_index=data.edge_index[:, val_mask],
        edge_attr=data.edge_attr[val_mask] if data.edge_attr is not None else None,
        y=data.y[val_mask] if data.y is not None else None,
        num_nodes=data.num_nodes
    )
    
    logger.info(f"Created train/val split: {train_mask.size(0)} train, {val_mask.size(0)} val edges")
    
    return train_data, val_data


if __name__ == "__main__":
    # Example training script
    torch.manual_seed(42)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load and preprocess data
    from utils.load_data import create_sample_data
    
    # Create sample data if needed
    if not os.path.exists("../data/train_transaction.csv"):
        create_sample_data("../data/", n_samples=5000)
    
    # Load data
    data_loader = IEEE_CIS_DataLoader("../data/")
    processed_data = data_loader.load_and_preprocess()
    
    # Create graph
    graph_data = create_graph_from_data(
        processed_data['full_data'], 
        graph_type='homogeneous'
    )
    graph_data = graph_data.to(device)
    
    # Create train/val split
    train_data, val_data = create_train_val_split(graph_data, val_ratio=0.2)
    
    # Create model
    model = create_fraud_gcn(
        node_feature_dim=graph_data.x.shape[1],
        edge_feature_dim=graph_data.edge_attr.shape[1] if graph_data.edge_attr is not None else 0,
        model_config={
            'hidden_dims': [128, 64, 32],
            'embedding_dim': 32,
            'dropout': 0.2
        }
    )
    
    # Setup trainer
    trainer = FraudGCNTrainer(
        model=model,
        device=device,
        experiment_name='fraud_gcn_demo'
    )
    
    # Compute class weights for imbalanced data
    class_weights = compute_class_weights(train_data.y)
    
    # Setup training
    trainer.setup_training(
        optimizer_config={'type': 'adam', 'lr': 0.001, 'weight_decay': 1e-5},
        scheduler_config={'type': 'reduce_on_plateau', 'patience': 5},
        loss_config={'type': 'weighted_focal', 'class_weights': class_weights, 'gamma': 2.0},
        early_stopping_config={'patience': 15}
    )
    
    # Train model
    history = trainer.train(
        train_data=train_data,
        val_data=val_data,
        num_epochs=50,
        save_every=10,
        log_every=1
    )
    
    # Plot training history
    trainer.plot_training_history("training_history.png")
    
    print("Training completed successfully!")
    print(f"Best metrics: {trainer.metric_tracker.best_metrics}") 