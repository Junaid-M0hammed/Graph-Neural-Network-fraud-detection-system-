"""
Traditional Machine Learning Models for Fraud Detection

This module implements classical ML approaches for comparison with GNN models:
- XGBoost with hyperparameter tuning
- Random Forest with balanced class weights
- Logistic Regression with regularization
- Support Vector Machine
- Ensemble methods combining multiple models
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           precision_recall_curve, roc_curve, average_precision_score)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import xgboost as xgb
from typing import Dict, List, Tuple, Optional, Any
import joblib
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudDetectionPreprocessor:
    """Preprocessing pipeline for fraud detection data"""
    
    def __init__(self):
        self.numeric_features = []
        self.categorical_features = []
        self.preprocessor = None
        self.label_encoders = {}
        self.is_fitted = False
        
    def identify_feature_types(self, df: pd.DataFrame) -> None:
        """Automatically identify numeric and categorical features"""
        # Exclude target and ID columns
        exclude_cols = ['TransactionID', 'isFraud']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        self.numeric_features = []
        self.categorical_features = []
        
        for col in feature_cols:
            if df[col].dtype in ['int64', 'float64']:
                self.numeric_features.append(col)
            else:
                self.categorical_features.append(col)
                
        logger.info(f"Identified {len(self.numeric_features)} numeric and {len(self.categorical_features)} categorical features")
    
    def create_preprocessor(self) -> None:
        """Create sklearn preprocessing pipeline"""
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', LabelEncoder())
        ])
        
        # Handle categorical features manually since LabelEncoder doesn't work in ColumnTransformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
            ],
            remainder='drop'
        )
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit preprocessor and transform data"""
        if not self.is_fitted:
            self.identify_feature_types(df)
            self.create_preprocessor()
            
            # Handle categorical features separately
            df_processed = df.copy()
            for col in self.categorical_features:
                if col in df_processed.columns:
                    le = LabelEncoder()
                    df_processed[col] = df_processed[col].fillna('missing').astype(str)
                    df_processed[col] = le.fit_transform(df_processed[col])
                    self.label_encoders[col] = le
                    self.numeric_features.append(col)  # Add to numeric after encoding
            
            # Recreate preprocessor with all features as numeric
            self.create_preprocessor()
            self.preprocessor.fit(df_processed[self.numeric_features])
            self.is_fitted = True
            
        return self.transform(df)
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted preprocessor"""
        df_processed = df.copy()
        
        # Apply label encoding to categorical features
        for col in self.categorical_features:
            if col in df_processed.columns and col in self.label_encoders:
                df_processed[col] = df_processed[col].fillna('missing').astype(str)
                # Handle unseen categories
                le = self.label_encoders[col]
                df_processed[col] = df_processed[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        
        return self.preprocessor.transform(df_processed[self.numeric_features])


class XGBoostFraudDetector:
    """XGBoost model optimized for fraud detection"""
    
    def __init__(self, **kwargs):
        self.params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 1,
            'random_state': 42,
            **kwargs
        }
        self.model = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Train XGBoost model"""
        # Calculate scale_pos_weight for imbalanced data
        neg_count = np.sum(y == 0)
        pos_count = np.sum(y == 1)
        self.params['scale_pos_weight'] = neg_count / pos_count
        
        self.model = xgb.XGBClassifier(**self.params)
        
        if X_val is not None and y_val is not None:
            try:
                self.model.fit(
                    X, y,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=10,
                    verbose=False
                )
            except TypeError:
                # Fallback for newer XGBoost versions
                self.model.fit(X, y)
        else:
            self.model.fit(X, y)
        
        self.is_fitted = True
        logger.info(f"XGBoost trained with scale_pos_weight: {self.params['scale_pos_weight']:.2f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if self.model:
            return dict(zip(range(len(self.model.feature_importances_)), self.model.feature_importances_))
        return {}


class RandomForestFraudDetector:
    """Random Forest model for fraud detection"""
    
    def __init__(self, **kwargs):
        self.params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1,
            **kwargs
        }
        self.model = RandomForestClassifier(**self.params)
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train Random Forest model"""
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info("Random Forest trained with balanced class weights")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        if self.model:
            return dict(zip(range(len(self.model.feature_importances_)), self.model.feature_importances_))
        return {}


class LogisticRegressionFraudDetector:
    """Logistic Regression with regularization for fraud detection"""
    
    def __init__(self, **kwargs):
        self.params = {
            'C': 1.0,
            'penalty': 'l2',
            'class_weight': 'balanced',
            'random_state': 42,
            'max_iter': 1000,
            **kwargs
        }
        self.model = LogisticRegression(**self.params)
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train Logistic Regression model"""
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info("Logistic Regression trained with balanced class weights")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        if self.model and hasattr(self.model, 'coef_'):
            return dict(zip(range(len(self.model.coef_[0])), np.abs(self.model.coef_[0])))
        return {}


class SVMFraudDetector:
    """Support Vector Machine for fraud detection"""
    
    def __init__(self, **kwargs):
        self.params = {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'class_weight': 'balanced',
            'probability': True,
            'random_state': 42,
            **kwargs
        }
        self.model = SVC(**self.params)
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train SVM model"""
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info("SVM trained with balanced class weights")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class EnsembleFraudDetector:
    """Ensemble of multiple fraud detection models"""
    
    def __init__(self, models: List[Any], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights or [1.0] * len(models)
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train all models in the ensemble"""
        for model in self.models:
            model.fit(X, y)
        self.is_fitted = True
        logger.info(f"Ensemble trained with {len(self.models)} models")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble predictions"""
        predictions = np.zeros((X.shape[0], 2))
        
        for model, weight in zip(self.models, self.weights):
            predictions += weight * model.predict_proba(X)
        
        # Normalize by sum of weights
        predictions /= sum(self.weights)
        return predictions
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get binary predictions"""
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)


class FraudDetectionComparison:
    """Compare multiple fraud detection models"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.preprocessor = FraudDetectionPreprocessor()
    
    def add_model(self, name: str, model: Any):
        """Add a model to comparison"""
        self.models[name] = model
    
    def load_data(self, transaction_path: str, identity_path: Optional[str] = None) -> pd.DataFrame:
        """Load and merge transaction and identity data"""
        df_transaction = pd.read_csv(transaction_path)
        
        if identity_path and Path(identity_path).exists():
            df_identity = pd.read_csv(identity_path)
            df = df_transaction.merge(df_identity, on='TransactionID', how='left')
            logger.info(f"Merged transaction and identity data: {len(df)} records")
        else:
            df = df_transaction
            logger.info(f"Using transaction data only: {len(df)} records")
        
        return df
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target"""
        X = self.preprocessor.fit_transform(df)
        y = df['isFraud'].values
        
        logger.info(f"Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Class distribution: {np.bincount(y)}")
        
        return X, y
    
    def train_and_evaluate(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict[str, Dict]:
        """Train all models and evaluate performance"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Further split training into train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            try:
                # Train model
                if isinstance(model, XGBoostFraudDetector):
                    model.fit(X_train, y_train, X_val, y_val)
                else:
                    model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                auc_score = roc_auc_score(y_test, y_proba)
                ap_score = average_precision_score(y_test, y_proba)
                
                results[name] = {
                    'auc': auc_score,
                    'average_precision': ap_score,
                    'predictions': y_pred,
                    'probabilities': y_proba,
                    'true_labels': y_test,
                    'feature_importance': model.get_feature_importance() if hasattr(model, 'get_feature_importance') else {}
                }
                
                logger.info(f"{name} - AUC: {auc_score:.4f}, AP: {ap_score:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                results[name] = {'error': str(e)}
        
        self.results = results
        return results
    
    def get_comparison_report(self) -> pd.DataFrame:
        """Generate comparison report"""
        report_data = []
        
        for name, result in self.results.items():
            if 'error' not in result:
                report_data.append({
                    'Model': name,
                    'ROC-AUC': result['auc'],
                    'Average Precision': result['average_precision'],
                    'Precision@0.5': np.mean(result['predictions'] == result['true_labels']),
                })
        
        return pd.DataFrame(report_data).sort_values('ROC-AUC', ascending=False)


def create_default_models() -> Dict[str, Any]:
    """Create default set of fraud detection models"""
    return {
        'XGBoost': XGBoostFraudDetector(),
        'Random Forest': RandomForestFraudDetector(),
        'Logistic Regression': LogisticRegressionFraudDetector(),
        'SVM': SVMFraudDetector()
    }


def create_ensemble_model() -> EnsembleFraudDetector:
    """Create ensemble of best performing models"""
    models = [
        XGBoostFraudDetector(),
        RandomForestFraudDetector(),
        LogisticRegressionFraudDetector()
    ]
    weights = [0.4, 0.3, 0.3]  # Give more weight to XGBoost
    return EnsembleFraudDetector(models, weights)


if __name__ == "__main__":
    # Example usage
    comparison = FraudDetectionComparison()
    
    # Add models
    for name, model in create_default_models().items():
        comparison.add_model(name, model)
    
    # Load and prepare data
    df = comparison.load_data("../data/train_transaction.csv", "../data/train_identity.csv")
    X, y = comparison.prepare_data(df)
    
    # Train and evaluate
    results = comparison.train_and_evaluate(X, y)
    
    # Print comparison
    report = comparison.get_comparison_report()
    print("\nModel Comparison Results:")
    print(report.to_string(index=False)) 