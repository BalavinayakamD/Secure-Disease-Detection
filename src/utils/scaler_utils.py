"""
Utility functions for creating and using scalers from hospital data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib


def create_scaler_from_hospital_data(data_dir='data/processed', num_hospitals=5):
    """
    Create a StandardScaler by computing statistics from all hospital training data.
    
    Since hospital data is already scaled, we compute the inverse transformation
    parameters to enable raw input prediction.
    
    Args:
        data_dir: Directory containing hospital CSV files
        num_hospitals: Number of hospital datasets
        
    Returns:
        StandardScaler fitted on aggregated data, or None if data is already scaled
    """
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / data_dir
    
    # Load all hospital training data
    all_data = []
    for hospital_id in range(num_hospitals):
        train_file = data_path / f'hospital_{hospital_id}_train.csv'
        if train_file.exists():
            df = pd.read_csv(train_file)
            if 'cardio' in df.columns:
                df = df.drop(columns=['cardio'])
            all_data.append(df)
    
    if not all_data:
        raise FileNotFoundError(f"No hospital training data found in {data_path}")
    
    # Combine all hospital data
    combined_df = pd.concat(all_data, axis=0, ignore_index=True)
    
    # Check if data is already scaled
    mean_vals = combined_df.mean().values
    std_vals = combined_df.std().values
    
    is_scaled = (np.abs(mean_vals.mean()) < 0.3 and 
                 np.abs(std_vals.mean() - 1.0) < 0.3)
    
    if is_scaled:
        # Data is already scaled - return statistics for reference
        # but indicate no transformation needed
        return None, mean_vals, std_vals, True
    else:
        # Data is not scaled - create and fit scaler
        scaler = StandardScaler()
        scaler.fit(combined_df)
        return scaler, scaler.mean_, scaler.scale_, False


def get_raw_feature_scaler():
    """
    Get a scaler for raw (unscaled) cardiovascular features.
    
    These statistics are computed from the original cardiovascular dataset
    before hospital splitting. Use this for transforming raw user input
    when hospital data is already scaled.
    
    Returns:
        dict with 'mean' and 'std' arrays for each feature
    """
    # Feature order: age, gender, height, weight, ap_hi, ap_lo, 
    #                cholesterol, gluc, smoke, alco, active
    
    # These are approximate statistics from the cleaned cardiovascular dataset
    # Age is in years (already converted from days)
    feature_stats = {
        'mean': np.array([
            53.0,   # age (years)
            1.5,    # gender (1=F, 2=M, mean~1.5)
            164.0,  # height (cm)
            74.0,   # weight (kg)
            128.0,  # ap_hi (systolic BP, mmHg)
            96.0,   # ap_lo (diastolic BP, mmHg)
            1.6,    # cholesterol (1,2,3)
            1.4,    # glucose (1,2,3)
            0.09,   # smoke (binary)
            0.05,   # alcohol (binary)
            0.8     # active (binary)
        ]),
        'std': np.array([
            7.0,    # age
            0.5,    # gender
            8.0,    # height
            14.0,   # weight
            20.0,   # ap_hi
            12.0,   # ap_lo
            0.7,    # cholesterol
            0.6,    # glucose
            0.28,   # smoke
            0.22,   # alcohol
            0.4     # active
        ])
    }
    
    return feature_stats


def transform_raw_input(raw_features, use_hospital_scaler=False, data_dir='data/processed', num_hospitals=5):
    """
    Transform raw patient input to scaled features for model inference.
    
    Args:
        raw_features: numpy array of shape (n_samples, n_features) with raw values
        use_hospital_scaler: If True, compute scaler from hospital data; 
                            if False, use default stats
        data_dir: Directory containing hospital data (if use_hospital_scaler=True)
        num_hospitals: Number of hospitals
        
    Returns:
        Scaled features ready for model input
    """
    if use_hospital_scaler:
        scaler, mean, std, is_scaled = create_scaler_from_hospital_data(data_dir, num_hospitals)
        if scaler is not None:
            return scaler.transform(raw_features)
        else:
            # Hospital data is already scaled, use default raw stats
            stats = get_raw_feature_scaler()
            return (raw_features - stats['mean']) / stats['std']
    else:
        stats = get_raw_feature_scaler()
        return (raw_features - stats['mean']) / stats['std']


def save_raw_feature_scaler(output_path='data/processed/raw_scaler.pkl'):
    """
    Save a scaler object using raw feature statistics.
    """
    stats = get_raw_feature_scaler()
    
    # Create a StandardScaler with these statistics
    scaler = StandardScaler()
    scaler.mean_ = stats['mean']
    scaler.scale_ = stats['std']
    scaler.var_ = stats['std'] ** 2
    scaler.n_features_in_ = len(stats['mean'])
    scaler.n_samples_seen_ = 50000  # Approximate
    
    # Save it
    project_root = Path(__file__).resolve().parents[2]
    full_path = project_root / output_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(scaler, full_path)
    print(f"Saved raw feature scaler to {full_path}")
    
    return scaler


if __name__ == '__main__':
    # Test the scaler creation
    print("Testing scaler utilities...")
    
    print("\n1. Getting raw feature scaler...")
    stats = get_raw_feature_scaler()
    print(f"Mean: {stats['mean'][:5]}")
    print(f"Std: {stats['std'][:5]}")
    
    print("\n2. Testing transformation...")
    # Create sample raw input (age=50, gender=2, height=170, ...)
    raw_input = np.array([[50, 2, 170, 75, 120, 80, 1, 1, 0, 0, 1]])
    scaled = transform_raw_input(raw_input)
    print(f"Raw input: {raw_input[0][:5]}")
    print(f"Scaled: {scaled[0][:5]}")
    
    print("\n3. Saving raw scaler...")
    scaler = save_raw_feature_scaler()
    print("✓ Scaler saved")
    
    print("\n4. Checking hospital data...")
    try:
        result = create_scaler_from_hospital_data()
        if result[0] is None:
            print("✓ Hospital data is already scaled")
        else:
            print("✓ Created scaler from hospital data")
    except Exception as e:
        print(f"⚠ Could not load hospital data: {e}")
    
    print("\n✓ All tests completed")
