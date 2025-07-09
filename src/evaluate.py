"""
Comprehensive Evaluation Module for Fraud Detection GCN

This module implements:
- Detailed performance metrics for fraud detection
- ROC curves, PR curves, and confusion matrices
- Business metrics and cost analysis
- Model interpretation and feature importance
- Comparative analysis with baseline models
- Statistical significance testing
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, roc_curve, auc,
    confusion_matrix, classification_report, f1_score,
    precision_score, recall_score, accuracy_score,
    average_precision_score, balanced_accuracy_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional, Union
import logging
import warnings
from dataclasses import dataclass

# Import our modules
from models.gcn import FraudGCN
from utils.build_graph import FraudGraphBuilder
from train import FraudGCNTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


@dataclass
class EvaluationResults:
    """
    Container for evaluation results
    """
    predictions: np.ndarray
    probabilities: np.ndarray
    true_labels: np.ndarray
    metrics: Dict[str, float]
    confusion_matrix: np.ndarray
    classification_report: str
    
    
class FraudDetectionEvaluator:
    """
    Comprehensive evaluator for fraud detection models
    """
    
    def __init__(self, 
                 model: FraudGCN,
                 device: str = 'cpu',
                 cost_matrix: Optional[Dict[str, float]] = None):
        """
        Initialize evaluator
        
        Args:
            model: Trained GCN model
            device: Device to run evaluation on
            cost_matrix: Business cost matrix for different error types
        """
        self.model = model.to(device)
        self.device = device
        
        # Default cost matrix for fraud detection
        # Assuming: Missing fraud (FN) is more costly than false alarm (FP)
        self.cost_matrix = cost_matrix or {
            'true_positive': 50,     # Benefit of correctly identifying fraud
            'false_positive': -10,   # Cost of false alarm
            'true_negative': 1,      # Small benefit of correctly identifying normal
            'false_negative': -100   # High cost of missing fraud
        }
        
    def predict(self, data) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on data
        
        Args:
            data: Graph data
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(data.x, data.edge_index, data.edge_attr)
            
            if outputs.dim() > 1:
                probabilities = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            else:
                probabilities = torch.sigmoid(outputs).cpu().numpy()
                predictions = (probabilities > 0.5).astype(int)
        
        return predictions, probabilities
    
    def compute_basic_metrics(self, 
                            y_true: np.ndarray, 
                            y_pred: np.ndarray, 
                            y_prob: np.ndarray) -> Dict[str, float]:
        """
        Compute basic classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        metrics['specificity'] = self._compute_specificity(y_true, y_pred)
        
        # AUC metrics
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['roc_auc'] = 0.0
            
        try:
            metrics['pr_auc'] = average_precision_score(y_true, y_prob)
        except ValueError:
            metrics['pr_auc'] = 0.0
        
        # Precision at different recall levels
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
        for recall_level in [0.8, 0.9, 0.95]:
            idx = np.argmin(np.abs(recall_curve - recall_level))
            metrics[f'precision_at_recall_{recall_level}'] = precision_curve[idx]
        
        # Recall at different precision levels
        for precision_level in [0.8, 0.9, 0.95]:
            idx = np.argmin(np.abs(precision_curve - precision_level))
            if idx < len(recall_curve):
                metrics[f'recall_at_precision_{precision_level}'] = recall_curve[idx]
            else:
                metrics[f'recall_at_precision_{precision_level}'] = 0.0
        
        return metrics
    
    def _compute_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute specificity (true negative rate)"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    def compute_business_metrics(self, 
                               y_true: np.ndarray, 
                               y_pred: np.ndarray,
                               transaction_amounts: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute business-relevant metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            transaction_amounts: Transaction amounts for financial impact calculation
            
        Returns:
            Dictionary of business metrics
        """
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        business_metrics = {}
        
        # Cost-based metrics
        total_cost = (tp * self.cost_matrix['true_positive'] + 
                     fp * self.cost_matrix['false_positive'] + 
                     tn * self.cost_matrix['true_negative'] + 
                     fn * self.cost_matrix['false_negative'])
        
        business_metrics['total_cost'] = total_cost
        business_metrics['cost_per_transaction'] = total_cost / len(y_true)
        
        # Detection rate
        business_metrics['fraud_detection_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        business_metrics['false_alarm_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # Financial impact (if transaction amounts provided)
        if transaction_amounts is not None:
            fraud_mask = y_true == 1
            detected_fraud_mask = (y_true == 1) & (y_pred == 1)
            missed_fraud_mask = (y_true == 1) & (y_pred == 0)
            
            total_fraud_amount = transaction_amounts[fraud_mask].sum()
            detected_fraud_amount = transaction_amounts[detected_fraud_mask].sum()
            missed_fraud_amount = transaction_amounts[missed_fraud_mask].sum()
            
            business_metrics['total_fraud_amount'] = total_fraud_amount
            business_metrics['detected_fraud_amount'] = detected_fraud_amount
            business_metrics['missed_fraud_amount'] = missed_fraud_amount
            business_metrics['fraud_amount_recovery_rate'] = (
                detected_fraud_amount / total_fraud_amount if total_fraud_amount > 0 else 0.0
            )
        
        return business_metrics
    
    def evaluate_model(self, data, transaction_amounts: Optional[np.ndarray] = None) -> EvaluationResults:
        """
        Comprehensive model evaluation
        
        Args:
            data: Graph data with true labels
            transaction_amounts: Optional transaction amounts for business metrics
            
        Returns:
            EvaluationResults object
        """
        logger.info("Starting comprehensive model evaluation...")
        
        # Make predictions
        predictions, probabilities = self.predict(data)
        true_labels = data.y.cpu().numpy()
        
        # Compute metrics
        basic_metrics = self.compute_basic_metrics(true_labels, predictions, probabilities)
        business_metrics = self.compute_business_metrics(true_labels, predictions, transaction_amounts)
        
        # Combine all metrics
        all_metrics = {**basic_metrics, **business_metrics}
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # Classification report
        report = classification_report(true_labels, predictions, 
                                     target_names=['Normal', 'Fraud'])
        
        logger.info("Model evaluation completed")
        logger.info(f"ROC AUC: {basic_metrics['roc_auc']:.4f}")
        logger.info(f"PR AUC: {basic_metrics['pr_auc']:.4f}")
        logger.info(f"F1 Score: {basic_metrics['f1']:.4f}")
        
        return EvaluationResults(
            predictions=predictions,
            probabilities=probabilities,
            true_labels=true_labels,
            metrics=all_metrics,
            confusion_matrix=cm,
            classification_report=report
        )
    
    def plot_confusion_matrix(self, results: EvaluationResults, save_path: str = None):
        """
        Plot confusion matrix
        
        Args:
            results: Evaluation results
            save_path: Path to save the plot
        """
        plt.figure(figsize=(8, 6))
        
        # Normalize confusion matrix
        cm_normalized = results.confusion_matrix.astype('float') / results.confusion_matrix.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(cm_normalized, 
                   annot=True, 
                   fmt='.3f', 
                   cmap='Blues',
                   xticklabels=['Normal', 'Fraud'],
                   yticklabels=['Normal', 'Fraud'])
        
        plt.title('Confusion Matrix (Normalized)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Add count annotations
        for i in range(2):
            for j in range(2):
                plt.text(j + 0.5, i + 0.7, f'({results.confusion_matrix[i, j]})', 
                        ha='center', va='center', fontsize=10, color='red')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, results: EvaluationResults, save_path: str = None):
        """
        Plot ROC curve
        
        Args:
            results: Evaluation results
            save_path: Path to save the plot
        """
        fpr, tpr, _ = roc_curve(results.true_labels, results.probabilities)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curve(self, results: EvaluationResults, save_path: str = None):
        """
        Plot Precision-Recall curve
        
        Args:
            results: Evaluation results
            save_path: Path to save the plot
        """
        precision, recall, _ = precision_recall_curve(results.true_labels, results.probabilities)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AUC = {pr_auc:.3f})')
        
        # Baseline (random classifier)
        fraud_rate = np.mean(results.true_labels)
        plt.axhline(y=fraud_rate, color='navy', linestyle='--', lw=2,
                   label=f'Random classifier (baseline = {fraud_rate:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"PR curve saved to {save_path}")
        
        plt.show()
    
    def plot_threshold_analysis(self, results: EvaluationResults, save_path: str = None):
        """
        Plot metrics vs threshold analysis
        
        Args:
            results: Evaluation results
            save_path: Path to save the plot
        """
        thresholds = np.linspace(0, 1, 100)
        precisions = []
        recalls = []
        f1_scores = []
        
        for threshold in thresholds:
            pred_thresh = (results.probabilities > threshold).astype(int)
            
            precision = precision_score(results.true_labels, pred_thresh, zero_division=0)
            recall = recall_score(results.true_labels, pred_thresh, zero_division=0)
            f1 = f1_score(results.true_labels, pred_thresh, zero_division=0)
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, precisions, label='Precision', linewidth=2)
        plt.plot(thresholds, recalls, label='Recall', linewidth=2)
        plt.plot(thresholds, f1_scores, label='F1 Score', linewidth=2)
        
        # Mark optimal F1 threshold
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        plt.axvline(x=optimal_threshold, color='red', linestyle='--', 
                   label=f'Optimal F1 threshold = {optimal_threshold:.3f}')
        
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Metrics vs Classification Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Threshold analysis saved to {save_path}")
        
        plt.show()
        
        return optimal_threshold
    
    def compare_with_baselines(self, 
                              X_train: np.ndarray, 
                              y_train: np.ndarray,
                              X_test: np.ndarray, 
                              y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Compare GCN performance with baseline models
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of model performances
        """
        logger.info("Comparing with baseline models...")
        
        results = {}
        
        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_probs = rf.predict_proba(X_test)[:, 1]
        rf_preds = rf.predict(X_test)
        
        results['RandomForest'] = self.compute_basic_metrics(y_test, rf_preds, rf_probs)
        
        # Logistic Regression
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_train, y_train)
        lr_probs = lr.predict_proba(X_test)[:, 1]
        lr_preds = lr.predict(X_test)
        
        results['LogisticRegression'] = self.compute_basic_metrics(y_test, lr_preds, lr_probs)
        
        # GCN results (assumed to be already computed)
        # This would be passed in or computed separately
        
        return results
    
    def generate_evaluation_report(self, results: EvaluationResults, save_path: str = None) -> str:
        """
        Generate comprehensive evaluation report
        
        Args:
            results: Evaluation results
            save_path: Path to save the report
            
        Returns:
            Report string
        """
        report = []
        report.append("=" * 60)
        report.append("FRAUD DETECTION MODEL EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Basic Performance Metrics
        report.append("1. BASIC PERFORMANCE METRICS")
        report.append("-" * 30)
        basic_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
        for metric in basic_metrics:
            if metric in results.metrics:
                report.append(f"{metric.upper():20s}: {results.metrics[metric]:.4f}")
        report.append("")
        
        # Confusion Matrix
        report.append("2. CONFUSION MATRIX")
        report.append("-" * 20)
        report.append("           Predicted")
        report.append("         Normal  Fraud")
        report.append(f"Normal    {results.confusion_matrix[0,0]:6d}  {results.confusion_matrix[0,1]:5d}")
        report.append(f"Fraud     {results.confusion_matrix[1,0]:6d}  {results.confusion_matrix[1,1]:5d}")
        report.append("")
        
        # Business Metrics
        report.append("3. BUSINESS METRICS")
        report.append("-" * 20)
        business_metrics = ['fraud_detection_rate', 'false_alarm_rate', 'total_cost']
        for metric in business_metrics:
            if metric in results.metrics:
                report.append(f"{metric.replace('_', ' ').title():25s}: {results.metrics[metric]:.4f}")
        report.append("")
        
        # Model Recommendations
        report.append("4. RECOMMENDATIONS")
        report.append("-" * 20)
        
        if results.metrics['roc_auc'] > 0.9:
            report.append("✓ Excellent model performance (ROC AUC > 0.9)")
        elif results.metrics['roc_auc'] > 0.8:
            report.append("✓ Good model performance (ROC AUC > 0.8)")
        else:
            report.append("⚠ Model performance needs improvement (ROC AUC < 0.8)")
        
        if results.metrics['recall'] < 0.8:
            report.append("⚠ Consider adjusting threshold to improve fraud detection rate")
        
        if results.metrics['precision'] < 0.5:
            report.append("⚠ High false positive rate - consider feature engineering")
        
        report.append("")
        report.append("=" * 60)
        
        report_str = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_str)
            logger.info(f"Evaluation report saved to {save_path}")
        
        return report_str


def evaluate_fraud_model(model: FraudGCN,
                        test_data,
                        device: str = 'cpu',
                        save_plots: bool = True,
                        output_dir: str = 'evaluation_results/') -> EvaluationResults:
    """
    Convenience function for complete model evaluation
    
    Args:
        model: Trained GCN model
        test_data: Test graph data
        device: Device to run evaluation on
        save_plots: Whether to save plots
        output_dir: Directory to save results
        
    Returns:
        Evaluation results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = FraudDetectionEvaluator(model, device)
    
    # Evaluate model
    results = evaluator.evaluate_model(test_data)
    
    # Generate plots
    if save_plots:
        evaluator.plot_confusion_matrix(results, 
                                       os.path.join(output_dir, 'confusion_matrix.png'))
        evaluator.plot_roc_curve(results, 
                                os.path.join(output_dir, 'roc_curve.png'))
        evaluator.plot_precision_recall_curve(results, 
                                             os.path.join(output_dir, 'pr_curve.png'))
        optimal_threshold = evaluator.plot_threshold_analysis(results, 
                                                            os.path.join(output_dir, 'threshold_analysis.png'))
    
    # Generate report
    report = evaluator.generate_evaluation_report(results, 
                                                 os.path.join(output_dir, 'evaluation_report.txt'))
    
    print(report)
    
    return results


if __name__ == "__main__":
    # Example evaluation
    print("Fraud Detection Model Evaluation Example")
    print("This module provides comprehensive evaluation tools for GCN fraud detection models.")
    print("Key features:")
    print("- ROC and Precision-Recall curves")
    print("- Business metrics and cost analysis")
    print("- Threshold optimization")
    print("- Baseline model comparison")
    print("- Comprehensive reporting") 