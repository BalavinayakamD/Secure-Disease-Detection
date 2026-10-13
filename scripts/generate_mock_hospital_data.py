"""
Generate minimal mock hospital data for testing and Live Demo.

Creates small hospital-specific train/test CSV files matching the expected format.
Used when full preprocessing hasn't been run yet.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# Get project root
project_root = Path(__file__).resolve().parents[1]
output_dir = project_root / 'data' / 'processed'

# Feature columns (11 features matching config)
feature_cols = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
                'cholesterol', 'gluc', 'smoke', 'alco', 'active']


def generate_hospital_data(hospital_id, n_train=200, n_test=50, positive_rate=0.5, seed=42):
    """Generate mock data for one hospital with specified class distribution."""
    rng = np.random.default_rng(seed + hospital_id)
    
    # Generate features (standardized, matching preprocessed format)
    def make_features(n):
        return pd.DataFrame({
            'age': rng.normal(0, 1, n),  # Standardized age
            'gender': rng.choice([1, 2], n),
            'height': rng.normal(0, 1, n),  # Standardized height
            'weight': rng.normal(0, 1, n),  # Standardized weight
            'ap_hi': rng.normal(0, 1, n),  # Standardized blood pressure
            'ap_lo': rng.normal(0, 1, n),
            'cholesterol': rng.choice([1, 2, 3], n),
            'gluc': rng.choice([1, 2, 3], n),
            'smoke': rng.choice([0, 1], n),
            'alco': rng.choice([0, 1], n),
            'active': rng.choice([0, 1], n)
        })
    
    # Training data
    train_features = make_features(n_train)
    n_positive_train = int(n_train * positive_rate)
    train_labels = np.array([1] * n_positive_train + [0] * (n_train - n_positive_train))
    rng.shuffle(train_labels)
    train_features['cardio'] = train_labels
    
    # Test data
    test_features = make_features(n_test)
    n_positive_test = int(n_test * positive_rate)
    test_labels = np.array([1] * n_positive_test + [0] * (n_test - n_positive_test))
    rng.shuffle(test_labels)
    test_features['cardio'] = test_labels
    
    return train_features, test_features


def main():
    """Generate mock hospital data for 3 hospitals with different distributions."""
    print("Generating mock hospital data...")
    print(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Hospital configurations matching expected non-IID distributions
    # Based on actual experiment results: hospital_0: 41%, hospital_1: 83%, hospital_2: 31%
    hospitals = [
        {'id': 0, 'positive_rate': 0.41, 'n_train': 300, 'n_test': 75},
        {'id': 1, 'positive_rate': 0.83, 'n_train': 250, 'n_test': 60},
        {'id': 2, 'positive_rate': 0.31, 'n_train': 280, 'n_test': 70},
    ]
    
    for hospital in hospitals:
        train_df, test_df = generate_hospital_data(
            hospital_id=hospital['id'],
            n_train=hospital['n_train'],
            n_test=hospital['n_test'],
            positive_rate=hospital['positive_rate']
        )
        
        train_path = output_dir / f"hospital_{hospital['id']}_train.csv"
        test_path = output_dir / f"hospital_{hospital['id']}_test.csv"
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"\nHospital {hospital['id']}:")
        print(f"  Train: {len(train_df)} samples ({train_df['cardio'].mean():.1%} positive)")
        print(f"  Test: {len(test_df)} samples ({test_df['cardio'].mean():.1%} positive)")
        print(f"  Saved to:")
        print(f"    {train_path}")
        print(f"    {test_path}")
    
    print(f"\n✓ Mock hospital data generated successfully")
    print(f"Total samples: {sum(h['n_train'] + h['n_test'] for h in hospitals)}")


if __name__ == '__main__':
    main()
