"""
Data Loading and Preprocessing Utilities for IEEE-CIS Fraud Detection

This module handles:
- Loading transaction and identity data
- Data cleaning and preprocessing
- Feature engineering
- Categorical encoding
- Data normalization and scaling
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List, Optional
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


class IEEE_CIS_DataLoader:
    """
    Comprehensive data loader for IEEE-CIS Fraud Detection dataset
    """
    
    def __init__(self, data_path: str = "data/"):
        """
        Initialize the data loader
        
        Args:
            data_path: Path to the directory containing the CSV files
        """
        self.data_path = data_path
        self.label_encoders = {}
        self.scaler = None
        self.feature_columns = []
        self.categorical_columns = []
        self.numerical_columns = []
        
    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load raw transaction and identity data from CSV files
        
        Returns:
            Tuple of (transaction_df, identity_df)
        """
        logger.info("Loading raw data files...")
        
        # Load transaction data
        train_transaction_path = os.path.join(self.data_path, "train_transaction.csv")
        train_identity_path = os.path.join(self.data_path, "train_identity.csv")
        
        if not os.path.exists(train_transaction_path):
            raise FileNotFoundError(f"Transaction file not found: {train_transaction_path}")
        if not os.path.exists(train_identity_path):
            raise FileNotFoundError(f"Identity file not found: {train_identity_path}")
            
        transaction_df = pd.read_csv(train_transaction_path)
        identity_df = pd.read_csv(train_identity_path)
        
        logger.info(f"Loaded transaction data: {transaction_df.shape}")
        logger.info(f"Loaded identity data: {identity_df.shape}")
        
        return transaction_df, identity_df
    
    def merge_data(self, transaction_df: pd.DataFrame, identity_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge transaction and identity data on TransactionID
        
        Args:
            transaction_df: Transaction data
            identity_df: Identity data
            
        Returns:
            Merged dataframe
        """
        logger.info("Merging transaction and identity data...")
        
        # Merge on TransactionID with left join to keep all transactions
        merged_df = transaction_df.merge(identity_df, on='TransactionID', how='left')
        
        logger.info(f"Merged data shape: {merged_df.shape}")
        
        return merged_df
    
    def identify_column_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Identify categorical and numerical columns
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary with 'categorical' and 'numerical' column lists
        """
        categorical_cols = []
        numerical_cols = []
        
        for col in df.columns:
            if col in ['TransactionID', 'isFraud']:
                continue
                
            # Check if column is categorical
            if (df[col].dtype == 'object' or 
                col.startswith('card') or 
                col.startswith('addr') or 
                col.endswith('domain') or
                col.startswith('M') or
                col.startswith('DeviceType') or
                col.startswith('DeviceInfo') or
                col.startswith('id_')):
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        
        self.categorical_columns = categorical_cols
        self.numerical_columns = numerical_cols
        
        logger.info(f"Identified {len(categorical_cols)} categorical and {len(numerical_cols)} numerical columns")
        
        return {
            'categorical': categorical_cols,
            'numerical': numerical_cols
        }
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with handled missing values
        """
        logger.info("Handling missing values...")
        
        df_cleaned = df.copy()
        
        # For categorical columns, fill with 'unknown'
        for col in self.categorical_columns:
            if col in df_cleaned.columns:
                df_cleaned[col] = df_cleaned[col].fillna('unknown')
        
        # For numerical columns, fill with median
        for col in self.numerical_columns:
            if col in df_cleaned.columns:
                median_val = df_cleaned[col].median()
                df_cleaned[col] = df_cleaned[col].fillna(median_val)
        
        logger.info("Missing values handled")
        
        return df_cleaned
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features using Label Encoding
        
        Args:
            df: Input dataframe
            fit: Whether to fit new encoders or use existing ones
            
        Returns:
            Dataframe with encoded categorical features
        """
        logger.info("Encoding categorical features...")
        
        df_encoded = df.copy()
        
        for col in self.categorical_columns:
            if col in df_encoded.columns:
                if fit:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        known_categories = set(self.label_encoders[col].classes_)
                        df_encoded[col] = df_encoded[col].astype(str)
                        unknown_mask = ~df_encoded[col].isin(known_categories)
                        df_encoded[col] = df_encoded[col].apply(
                            lambda x: x if x in known_categories else 'unknown'
                        )
                        df_encoded[col] = self.label_encoders[col].transform(df_encoded[col])
        
        logger.info("Categorical encoding completed")
        
        return df_encoded
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform feature engineering on the dataset
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with engineered features
        """
        logger.info("Performing feature engineering...")
        
        df_engineered = df.copy()
        
        # Time-based features
        if 'TransactionDT' in df_engineered.columns:
            # Convert to hour of day, day of week, etc.
            df_engineered['TransactionHour'] = (df_engineered['TransactionDT'] / 3600) % 24
            df_engineered['TransactionDayOfWeek'] = (df_engineered['TransactionDT'] / (3600 * 24)) % 7
            
        # Amount-based features
        if 'TransactionAmt' in df_engineered.columns:
            # Log transform for amount (add 1 to handle zeros)
            df_engineered['TransactionAmt_log'] = np.log1p(df_engineered['TransactionAmt'])
            
            # Amount percentile features
            df_engineered['TransactionAmt_percentile'] = df_engineered['TransactionAmt'].rank(pct=True)
        
        # Card and address frequency features
        for col in ['card1', 'card2', 'addr1', 'addr2']:
            if col in df_engineered.columns:
                freq_col = f'{col}_freq'
                df_engineered[freq_col] = df_engineered.groupby(col)[col].transform('count')
        
        # Email domain features
        for col in ['P_emaildomain', 'R_emaildomain']:
            if col in df_engineered.columns:
                freq_col = f'{col}_freq'
                df_engineered[freq_col] = df_engineered.groupby(col)[col].transform('count')
        
        logger.info("Feature engineering completed")
        
        return df_engineered
    
    def normalize_numerical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Normalize numerical features using RobustScaler
        
        Args:
            df: Input dataframe
            fit: Whether to fit new scaler or use existing one
            
        Returns:
            Dataframe with normalized numerical features
        """
        logger.info("Normalizing numerical features...")
        
        df_normalized = df.copy()
        
        # Get current numerical columns (including engineered ones)
        current_numerical_cols = [col for col in df_normalized.columns 
                                if col not in self.categorical_columns and 
                                col not in ['TransactionID', 'isFraud'] and
                                df_normalized[col].dtype in ['int64', 'float64']]
        
        if fit:
            self.scaler = RobustScaler()
            df_normalized[current_numerical_cols] = self.scaler.fit_transform(
                df_normalized[current_numerical_cols]
            )
        else:
            if self.scaler is not None:
                df_normalized[current_numerical_cols] = self.scaler.transform(
                    df_normalized[current_numerical_cols]
                )
        
        self.numerical_columns = current_numerical_cols
        
        logger.info("Numerical normalization completed")
        
        return df_normalized
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare final feature matrix and target vector
        
        Args:
            df: Preprocessed dataframe
            
        Returns:
            Tuple of (features_df, target_array)
        """
        logger.info("Preparing final features...")
        
        # Drop ID column and extract target
        features_df = df.drop(['TransactionID', 'isFraud'], axis=1, errors='ignore')
        target = df['isFraud'].values if 'isFraud' in df.columns else None
        
        self.feature_columns = list(features_df.columns)
        
        logger.info(f"Final feature matrix shape: {features_df.shape}")
        if target is not None:
            logger.info(f"Target distribution: {pd.Series(target).value_counts().to_dict()}")
        
        return features_df, target
    
    def load_and_preprocess(self, test_size: float = 0.2, random_state: int = 42) -> Dict:
        """
        Complete data loading and preprocessing pipeline
        
        Args:
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing all processed data splits
        """
        logger.info("Starting complete data preprocessing pipeline...")
        
        # Load raw data
        transaction_df, identity_df = self.load_raw_data()
        
        # Merge data
        merged_df = self.merge_data(transaction_df, identity_df)
        
        # Identify column types
        self.identify_column_types(merged_df)
        
        # Handle missing values
        cleaned_df = self.handle_missing_values(merged_df)
        
        # Encode categorical features
        encoded_df = self.encode_categorical_features(cleaned_df, fit=True)
        
        # Feature engineering
        engineered_df = self.feature_engineering(encoded_df)
        
        # Normalize numerical features
        normalized_df = self.normalize_numerical_features(engineered_df, fit=True)
        
        # Prepare features and target
        features_df, target = self.prepare_features(normalized_df)
        
        # Train-test split
        if target is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                features_df, target, test_size=test_size, 
                random_state=random_state, stratify=target
            )
            
            logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
            
            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'feature_columns': self.feature_columns,
                'categorical_columns': self.categorical_columns,
                'numerical_columns': self.numerical_columns,
                'full_data': normalized_df
            }
        else:
            return {
                'features': features_df,
                'feature_columns': self.feature_columns,
                'categorical_columns': self.categorical_columns,
                'numerical_columns': self.numerical_columns,
                'full_data': normalized_df
            }


def create_sample_data(save_path: str = "data/", n_samples: int = 10000) -> None:
    """
    Create sample data for testing when real dataset is not available
    
    Args:
        save_path: Path to save sample data
        n_samples: Number of samples to generate
    """
    logger.info(f"Creating sample data with {n_samples} samples...")
    
    np.random.seed(42)
    
    # Create sample transaction data
    transaction_data = {
        'TransactionID': range(1, n_samples + 1),
        'isFraud': np.random.choice([0, 1], n_samples, p=[0.96, 0.04]),  # 4% fraud rate
        'TransactionDT': np.random.randint(0, 1000000, n_samples),
        'TransactionAmt': np.random.lognormal(3, 1, n_samples),
        'ProductCD': np.random.choice(['W', 'H', 'C', 'S', 'R'], n_samples),
        'card1': np.random.randint(1000, 20000, n_samples),
        'card2': np.random.choice([100, 200, 300, 400, 500, np.nan], n_samples),
        'card3': np.random.choice([150, 185, np.nan], n_samples),
        'card4': np.random.choice(['visa', 'mastercard', 'american express', 'discover', np.nan], n_samples),
        'card5': np.random.choice([100, 101, 102, 200, 201, np.nan], n_samples),
        'card6': np.random.choice(['debit', 'credit', np.nan], n_samples),
        'addr1': np.random.randint(100, 500, n_samples),
        'addr2': np.random.randint(10, 100, n_samples),
        'P_emaildomain': np.random.choice(['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', np.nan], n_samples),
        'R_emaildomain': np.random.choice(['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', np.nan], n_samples),
    }
    
    # Add some numerical features
    for i in range(1, 10):
        transaction_data[f'C{i}'] = np.random.randn(n_samples)
        transaction_data[f'D{i}'] = np.random.randint(0, 100, n_samples)
        transaction_data[f'V{i}'] = np.random.randn(n_samples)
    
    # Create sample identity data (subset of transactions)
    identity_sample_size = n_samples // 3
    identity_data = {
        'TransactionID': np.random.choice(range(1, n_samples + 1), identity_sample_size, replace=False),
        'DeviceType': np.random.choice(['desktop', 'mobile', np.nan], identity_sample_size),
        'DeviceInfo': np.random.choice(['Windows', 'iOS', 'MacOS', 'Android', np.nan], identity_sample_size),
    }
    
    # Add identity features
    for i in range(1, 15):
        identity_data[f'id_{i:02d}'] = np.random.choice([0, 1, 2, np.nan], identity_sample_size)
    
    # Create DataFrames and save
    os.makedirs(save_path, exist_ok=True)
    
    transaction_df = pd.DataFrame(transaction_data)
    identity_df = pd.DataFrame(identity_data)
    
    transaction_df.to_csv(os.path.join(save_path, 'train_transaction.csv'), index=False)
    identity_df.to_csv(os.path.join(save_path, 'train_identity.csv'), index=False)
    
    logger.info(f"Sample data saved to {save_path}")


if __name__ == "__main__":
    # Example usage
    loader = IEEE_CIS_DataLoader()
    
    # Create sample data if needed
    if not os.path.exists("data/train_transaction.csv"):
        create_sample_data()
    
    # Load and preprocess data
    data = loader.load_and_preprocess()
    
    print("Data preprocessing completed successfully!")
    print(f"Training features shape: {data['X_train'].shape}")
    print(f"Training target shape: {data['y_train'].shape}") 