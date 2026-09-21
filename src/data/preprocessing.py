"""
Utility functions for loading preprocessed cardiovascular disease data.
"""

import pandas as pd
import joblib
import os


def load_processed_data(data_dir='data/processed'):
    """
    Load preprocessed train and test data.
    
    Args:
        data_dir: Directory containing processed data files
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    X_train = train.drop(columns=['cardio'])
    y_train = train['cardio']
    X_test = test.drop(columns=['cardio'])
    y_test = test['cardio']
    
    return X_train, X_test, y_train, y_test


def load_scaler(data_dir='data/processed'):
    """
    Load the fitted StandardScaler.
    
    Args:
        data_dir: Directory containing the scaler file
        
    Returns:
        StandardScaler: Fitted scaler object
    """
    scaler_path = os.path.join(data_dir, 'scaler.pkl')
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
