#!/usr/bin/env python3
"""
Data Setup Script for GNN Fraud Detection System
Downloads IEEE-CIS dataset from Kaggle or creates sample data for testing
"""
import os
import sys
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path

def create_data_directory():
    """Create data directory if it doesn't exist"""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    return data_dir

def check_kaggle_setup():
    """Check if Kaggle API is properly configured"""
    try:
        import kaggle
        # Test kaggle API
        subprocess.run(['kaggle', '--version'], capture_output=True, check=True)
        return True
    except (ImportError, subprocess.CalledProcessError, FileNotFoundError):
        return False

def download_kaggle_dataset(data_dir):
    """Download IEEE-CIS dataset from Kaggle"""
    print("Downloading IEEE-CIS Fraud Detection dataset from Kaggle...")
    
    try:
        # Download competition data
        cmd = ['kaggle', 'competitions', 'download', '-c', 'ieee-fraud-detection', '-p', str(data_dir)]
        subprocess.run(cmd, check=True)
        
        # Extract files
        import zipfile
        zip_path = data_dir / "ieee-fraud-detection.zip"
        
        if zip_path.exists():
            print("Extracting dataset files...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            
            # Remove zip file
            zip_path.unlink()
            
            print("Dataset downloaded and extracted successfully!")
            return True
        else:
            print("Error: Downloaded file not found")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        return False
    except Exception as e:
        print(f"Error extracting dataset: {e}")
        return False

def create_sample_data(data_dir, n_samples=10000):
    """Create synthetic sample data for testing"""
    print(f"Creating sample transaction data ({n_samples:,} transactions)...")
    
    np.random.seed(42)
    
    # Generate transaction data
    transaction_data = {
        'TransactionID': range(1, n_samples + 1),
        'isFraud': np.random.choice([0, 1], n_samples, p=[0.965, 0.035]),
        'TransactionDT': np.random.randint(86400, 15552000, n_samples),  # Time in seconds
        'TransactionAmt': np.random.lognormal(3, 1, n_samples),
        
        # Card features
        'card1': np.random.randint(1000, 20000, n_samples),
        'card2': np.random.choice([100, 111, 120, 150, 200], n_samples),
        'card3': np.random.choice([100, 150, 185], n_samples),
        'card4': np.random.choice(['visa', 'mastercard', 'american express', 'discover'], n_samples),
        'card5': np.random.choice([100, 102, 111, 117, 119, 142, 150], n_samples),
        'card6': np.random.choice(['credit', 'debit'], n_samples),
        
        # Address features
        'addr1': np.random.randint(100, 500, n_samples),
        'addr2': np.random.randint(10, 100, n_samples),
        
        # Distance features
        'dist1': np.random.exponential(50, n_samples),
        'dist2': np.random.exponential(100, n_samples),
        
        # Email domain features
        'P_emaildomain': np.random.choice([
            'gmail.com', 'yahoo.com', 'hotmail.com', 'anonymous.com',
            'outlook.com', 'icloud.com', 'aol.com', None
        ], n_samples, p=[0.3, 0.15, 0.15, 0.05, 0.1, 0.05, 0.05, 0.15]),
        
        'R_emaildomain': np.random.choice([
            'gmail.com', 'yahoo.com', 'hotmail.com', 'anonymous.com',
            'outlook.com', 'icloud.com', 'aol.com', None
        ], n_samples, p=[0.25, 0.1, 0.1, 0.05, 0.1, 0.05, 0.05, 0.3]),
        
        # Product features
        'ProductCD': np.random.choice(['W', 'C', 'R', 'H', 'S'], n_samples, p=[0.6, 0.15, 0.1, 0.1, 0.05]),
        
        # Device features
        'DeviceType': np.random.choice(['desktop', 'mobile'], n_samples, p=[0.4, 0.6]),
        'DeviceInfo': np.random.choice([
            'Windows', 'iOS Device', 'MacOS', 'Android', 'Linux'
        ], n_samples, p=[0.4, 0.25, 0.15, 0.15, 0.05])
    }
    
    # Add some M features (anonymized)
    for i in range(1, 10):
        transaction_data[f'M{i}'] = np.random.choice(['T', 'F', None], n_samples, p=[0.3, 0.3, 0.4])
    
    # Add some V features (Vesta engineered)
    for i in range(1, 21):
        transaction_data[f'V{i}'] = np.random.normal(0, 1, n_samples)
    
    # Create DataFrame
    train_transaction = pd.DataFrame(transaction_data)
    
    # Create identity data
    n_identity = int(n_samples * 0.3)  # Not all transactions have identity
    identity_ids = np.random.choice(train_transaction['TransactionID'], n_identity, replace=False)
    
    identity_data = {
        'TransactionID': identity_ids,
        'id_01': np.random.normal(0, 1, n_identity),
        'id_02': np.random.normal(0, 1, n_identity),
        'id_03': np.random.normal(0, 1, n_identity),
        'id_04': np.random.normal(0, 1, n_identity),
        'id_05': np.random.normal(0, 1, n_identity),
        'DeviceType': np.random.choice(['desktop', 'mobile'], n_identity),
        'DeviceInfo': np.random.choice([
            'Windows 10', 'iOS 13', 'Android 9', 'MacOS 10.15'
        ], n_identity)
    }
    
    train_identity = pd.DataFrame(identity_data)
    
    # Save to CSV files
    train_transaction.to_csv(data_dir / "train_transaction.csv", index=False)
    train_identity.to_csv(data_dir / "train_identity.csv", index=False)
    
    print(f"Sample data created:")
    print(f"  - train_transaction.csv: {len(train_transaction):,} transactions")
    print(f"  - train_identity.csv: {len(train_identity):,} identity records")
    print(f"  - Fraud rate: {train_transaction['isFraud'].mean()*100:.1f}%")
    
    return True

def main():
    print("GNN Fraud Detection - Data Setup")
    print("=" * 40)
    
    # Create data directory
    data_dir = create_data_directory()
    
    # Check if data already exists
    if (data_dir / "train_transaction.csv").exists():
        response = input("Dataset already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Setup cancelled.")
            return
    
    print("\nData Source Options:")
    print("1. Download from Kaggle (requires Kaggle API setup)")
    print("2. Create sample data for testing")
    
    choice = input("\nSelect option (1 or 2): ").strip()
    
    if choice == "1":
        if check_kaggle_setup():
            print("\nKaggle API detected. Proceeding with download...")
            success = download_kaggle_dataset(data_dir)
        else:
            print("\nKaggle API not configured. Please:")
            print("1. Install kaggle: pip install kaggle")
            print("2. Set up API credentials: https://github.com/Kaggle/kaggle-api#api-credentials")
            print("3. Run this script again")
            
            fallback = input("\nCreate sample data instead? (y/N): ")
            if fallback.lower() == 'y':
                success = create_sample_data(data_dir)
            else:
                success = False
    
    elif choice == "2":
        n_samples = input("\nNumber of sample transactions (default 10000): ").strip()
        try:
            n_samples = int(n_samples) if n_samples else 10000
        except ValueError:
            n_samples = 10000
            
        success = create_sample_data(data_dir, n_samples)
    
    else:
        print("Invalid choice. Please run the script again.")
        return
    
    if success:
        print("\n" + "=" * 40)
        print("Data setup completed successfully!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run the dashboard: python run_dashboard.py")
        print("3. Or run training: python src/train.py")
    else:
        print("\nData setup failed. Please check the errors above.")

if __name__ == "__main__":
    main() 