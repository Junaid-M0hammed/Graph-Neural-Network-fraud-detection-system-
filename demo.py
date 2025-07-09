#!/usr/bin/env python3
"""
🔍 GNN Fraud Detection Demo Script

This script demonstrates the complete end-to-end pipeline for fraud detection
using Graph Neural Networks on the IEEE-CIS dataset.

Usage:
    python demo.py [--model gcn|gat] [--epochs 30] [--sample-size 5000]

Features:
- Data loading and preprocessing
- Graph construction from tabular data
- GCN/GAT model training
- Comprehensive evaluation
- Performance visualization
- Model comparison
"""

import argparse
import os
import sys
import time
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Imports
from utils.load_data import IEEE_CIS_DataLoader, create_sample_data
from utils.build_graph import create_graph_from_data
from models.gcn import create_fraud_gcn, compute_class_weights
from models.gat import create_fraud_gat, compare_gat_gcn_performance
from train import FraudGCNTrainer, create_train_val_split
from evaluate import evaluate_fraud_model

import warnings
warnings.filterwarnings('ignore')


def print_banner():
    """Print welcome banner"""
    print("🔍" + "="*60 + "🔍")
    print("🚀      GNN-BASED FRAUD DETECTION DEMO                 🚀")
    print("🔍" + "="*60 + "🔍")
    print("📊 Graph Neural Networks for FinTech Fraud Detection  📊")
    print("🧠 State-of-the-art Graph AI in Production            🧠")
    print("=" * 64)
    print()


def setup_environment():
    """Setup environment and check dependencies"""
    print("🔧 Setting up environment...")
    
    # Check PyTorch and PyG installation
    try:
        import torch
        import torch_geometric
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ PyTorch Geometric: {torch_geometric.__version__}")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install required packages: pip install -r requirements.txt")
        sys.exit(1)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Create directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    print("✅ Environment setup complete!\n")
    return device


def load_and_preprocess_data(sample_size: int = 5000):
    """Load and preprocess the fraud detection dataset"""
    print("📊 Loading and preprocessing data...")
    
    # Create sample data if needed
    data_path = "data/"
    if not os.path.exists(os.path.join(data_path, "train_transaction.csv")):
        print(f"📂 Creating sample IEEE-CIS data ({sample_size:,} transactions)...")
        create_sample_data(data_path, n_samples=sample_size)
    
    # Load and preprocess
    loader = IEEE_CIS_DataLoader(data_path)
    processed_data = loader.load_and_preprocess(test_size=0.2, random_state=42)
    
    fraud_rate = processed_data['y_train'].mean()
    print(f"✅ Data loaded: {processed_data['X_train'].shape[0]:,} train, {processed_data['X_test'].shape[0]:,} test")
    print(f"💳 Fraud rate: {fraud_rate:.2%}")
    print(f"📈 Features: {processed_data['X_train'].shape[1]}")
    print()
    
    return processed_data


def build_graph(processed_data):
    """Build graph structure from tabular data"""
    print("🕸️ Building graph structure...")
    
    # Create graph
    graph_data = create_graph_from_data(
        processed_data['full_data'], 
        graph_type='homogeneous',
        node_types=['card', 'addr', 'email']
    )
    
    print(f"✅ Graph created:")
    print(f"   🔗 Nodes: {graph_data.num_nodes:,}")
    print(f"   🔗 Edges: {graph_data.edge_index.shape[1]:,}")
    print(f"   📊 Node features: {graph_data.x.shape[1]}")
    print(f"   📊 Edge features: {graph_data.edge_attr.shape[1] if graph_data.edge_attr is not None else 0}")
    
    if graph_data.y is not None:
        edge_fraud_rate = graph_data.y.float().mean().item()
        print(f"   💳 Edge fraud rate: {edge_fraud_rate:.2%}")
    
    print()
    return graph_data


def train_model(graph_data, model_type: str = 'gcn', epochs: int = 30, device: str = 'cpu'):
    """Train GNN model"""
    print(f"🧠 Training {model_type.upper()} model...")
    
    # Move graph to device
    graph_data = graph_data.to(device)
    
    # Create train/val split
    train_data, val_data = create_train_val_split(graph_data, val_ratio=0.2, random_state=42)
    
    # Create model
    if model_type.lower() == 'gcn':
        model = create_fraud_gcn(
            node_feature_dim=graph_data.x.shape[1],
            edge_feature_dim=graph_data.edge_attr.shape[1] if graph_data.edge_attr is not None else 0,
            model_config={
                'hidden_dims': [128, 64, 32],
                'embedding_dim': 32,
                'dropout': 0.2
            }
        )
    elif model_type.lower() == 'gat':
        model = create_fraud_gat(
            node_feature_dim=graph_data.x.shape[1],
            edge_feature_dim=graph_data.edge_attr.shape[1] if graph_data.edge_attr is not None else 0,
            model_config={
                'hidden_dims': [64, 32],
                'heads': [8, 4, 1],
                'embedding_dim': 32,
                'dropout': 0.2
            }
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"🏗️ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Setup trainer
    trainer = FraudGCNTrainer(
        model=model,
        device=device,
        experiment_name=f'fraud_demo_{model_type}'
    )
    
    # Compute class weights
    class_weights = compute_class_weights(train_data.y)
    
    # Setup training
    trainer.setup_training(
        optimizer_config={'type': 'adam', 'lr': 0.001, 'weight_decay': 1e-5},
        scheduler_config={'type': 'reduce_on_plateau', 'patience': 5},
        loss_config={'type': 'weighted_focal', 'class_weights': class_weights, 'gamma': 2.0},
        early_stopping_config={'patience': 10}
    )
    
    # Train
    start_time = time.time()
    history = trainer.train(
        train_data=train_data,
        val_data=val_data,
        num_epochs=epochs,
        save_every=max(epochs // 3, 5),
        log_every=max(epochs // 10, 1)
    )
    
    training_time = time.time() - start_time
    best_auc = trainer.metric_tracker.best_metrics.get('roc_auc', 0)
    
    print(f"✅ Training completed in {training_time:.1f}s")
    print(f"🏆 Best validation ROC-AUC: {best_auc:.4f}")
    print()
    
    return model, trainer, val_data


def evaluate_model_performance(model, test_data, model_name: str, device: str = 'cpu'):
    """Evaluate model performance"""
    print(f"📈 Evaluating {model_name} performance...")
    
    # Run evaluation
    results = evaluate_fraud_model(
        model=model,
        test_data=test_data,
        device=device,
        save_plots=True,
        output_dir=f"results/{model_name.lower()}_evaluation/"
    )
    
    # Print key metrics
    print(f"✅ {model_name} Evaluation Results:")
    print(f"   🎯 ROC-AUC: {results.metrics['roc_auc']:.4f}")
    print(f"   🎯 PR-AUC: {results.metrics['pr_auc']:.4f}")
    print(f"   🎯 F1 Score: {results.metrics['f1']:.4f}")
    print(f"   🎯 Precision: {results.metrics['precision']:.4f}")
    print(f"   🎯 Recall: {results.metrics['recall']:.4f}")
    print(f"   💰 Fraud Detection Rate: {results.metrics.get('fraud_detection_rate', 0):.4f}")
    print()
    
    return results


def compare_models(gcn_model, gat_model, test_data, device: str = 'cpu'):
    """Compare GCN and GAT model performance"""
    print("🔍 Comparing GCN vs GAT performance...")
    
    try:
        comparison = compare_gat_gcn_performance(
            gat_model=gat_model,
            gcn_model=gcn_model,
            test_data=test_data,
            device=device
        )
        print("✅ Model comparison completed!")
        print()
    except Exception as e:
        print(f"⚠️ Model comparison failed: {e}")
        print()


def print_summary():
    """Print demo summary"""
    print("🎉" + "="*60 + "🎉")
    print("🎊           DEMO COMPLETED SUCCESSFULLY!            🎊")
    print("🎉" + "="*60 + "🎉")
    print()
    print("📋 What was demonstrated:")
    print("   ✅ Data preprocessing and feature engineering")
    print("   ✅ Graph construction from tabular fraud data")
    print("   ✅ GNN model training with advanced techniques")
    print("   ✅ Comprehensive fraud detection evaluation")
    print("   ✅ Performance visualization and reporting")
    print()
    print("📁 Generated outputs:")
    print("   📊 results/ - Evaluation plots and reports")
    print("   💾 checkpoints/ - Trained model checkpoints")
    print("   📈 Training history plots")
    print()
    print("🚀 Next steps:")
    print("   📓 Explore notebooks/fraud_detection_demo.ipynb")
    print("   🔧 Experiment with different model configurations")
    print("   📊 Try real IEEE-CIS dataset from Kaggle")
    print("   🚀 Deploy model with FastAPI (app/ directory)")
    print()
    print("💡 Why GNNs excel at fraud detection:")
    print("   🕸️ Captures relational patterns between entities")
    print("   🔗 Learns from shared cards, addresses, devices")
    print("   📈 Propagates fraud signals through graph structure")
    print("   🎯 Provides interpretable attention mechanisms")
    print()
    print("Thanks for exploring GNN-based fraud detection! 🙏")


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="GNN Fraud Detection Demo")
    parser.add_argument('--model', choices=['gcn', 'gat', 'both'], default='gcn',
                       help='Model type to train (default: gcn)')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs (default: 30)')
    parser.add_argument('--sample-size', type=int, default=5000,
                       help='Number of sample transactions to generate (default: 5000)')
    
    args = parser.parse_args()
    
    # Setup
    print_banner()
    device = setup_environment()
    
    try:
        # Data pipeline
        processed_data = load_and_preprocess_data(args.sample_size)
        graph_data = build_graph(processed_data)
        
        # Model training and evaluation
        if args.model in ['gcn', 'both']:
            gcn_model, gcn_trainer, val_data = train_model(
                graph_data, 'gcn', args.epochs, device
            )
            gcn_results = evaluate_model_performance(gcn_model, val_data, 'GCN', device)
        
        if args.model in ['gat', 'both']:
            gat_model, gat_trainer, val_data = train_model(
                graph_data, 'gat', args.epochs, device
            )
            gat_results = evaluate_model_performance(gat_model, val_data, 'GAT', device)
        
        # Model comparison
        if args.model == 'both':
            compare_models(gcn_model, gat_model, val_data, device)
        
        # Summary
        print_summary()
        
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 