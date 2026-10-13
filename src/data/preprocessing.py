"""
Utility functions for loading preprocessed cardiovascular disease data.
"""

import pandas as pd
import joblib
import os
from pathlib import Path


def _get_project_root():
    """Get project root directory."""
    return Path(__file__).resolve().parents[2]


def load_processed_data(data_dir='data/processed'):
    """
    Load preprocessed train and test data.
    
    Args:
        data_dir: Directory containing processed data files (relative to project root or absolute)
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Resolve path relative to project root if not absolute
    if not os.path.isabs(data_dir):
        data_dir = _get_project_root() / data_dir
    
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Processed train data not found at {train_path}.\n"
            f"Run data preprocessing first or check that hospital-specific data files exist."
        )
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Processed test data not found at {test_path}.\n"
            f"Run data preprocessing first or check that hospital-specific data files exist."
        )
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    required_cols = ['cardio']
    for col in required_cols:
        if col not in train.columns:
            raise ValueError(f"Required column '{col}' missing from {train_path}")
        if col not in test.columns:
            raise ValueError(f"Required column '{col}' missing from {test_path}")
    
    X_train = train.drop(columns=['cardio'])
    y_train = train['cardio']
    X_test = test.drop(columns=['cardio'])
    y_test = test['cardio']
    
    return X_train, X_test, y_train, y_test


def load_scaler(data_dir='data/processed'):
    """
    Load the fitted StandardScaler.
    
    Args:
        data_dir: Directory containing the scaler file (relative to project root or absolute)
        
    Returns:
        StandardScaler: Fitted scaler object
    """
    # Resolve path relative to project root if not absolute
    if not os.path.isabs(data_dir):
        data_dir = _get_project_root() / data_dir
    
    scaler_path = os.path.join(data_dir, 'scaler.pkl')
    
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    
    return joblib.load(scaler_path)


def get_feature_names():
    """
    Get the list of feature column names after preprocessing.
    
    Returns:
        list: Feature column names
    """
    return [
        'age',
        'gender', 
        'height',
        'weight',
        'ap_hi',
        'ap_lo',
        'cholesterol',
        'gluc',
        'smoke',
        'alco',
        'active'
    ]


if __name__ == '__main__':
    # Test loading
    print("Testing data loading...")
    X_train, X_test, y_train, y_test = load_processed_data()
    print(f"✓ Loaded train: {X_train.shape}, test: {X_test.shape}")
    
    scaler = load_scaler()
    print(f"✓ Loaded scaler: {type(scaler).__name__}")
    
    features = get_feature_names()
    print(f"✓ Feature names ({len(features)}): {features}")
