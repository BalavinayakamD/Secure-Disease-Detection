"""
Complete preprocessing pipeline for cardiovascular disease dataset.
Handles data cleaning, validation, encoding, scaling, and train/test split.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


class CardioPreprocessor:
    """
    Preprocessing pipeline for cardiovascular disease dataset.
    
    Steps:
    1. Load raw data
    2. Handle invalid values
    3. Remove ID column
    4. Separate features and target
    5. Train/test split (stratified)
    6. Fit scaler on training data only
    7. Transform train and test sets
    8. Save processed data and scaler
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = 'cardio'
        self.removed_rows = 0
        self.invalid_values_found = {}
        self.duplicate_rows = 0
        self.conflicting_rows = 0
        
    def load_data(self, filepath):
        """Load raw CSV data."""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath, sep=';')
        print(f"Initial shape: {df.shape}")
        return df
    
    def handle_invalid_values(self, df):
        """
        Identify and handle invalid values.
        
        Invalid criteria:
        - Height: < 100 cm or > 250 cm (clearly data entry errors)
        - Weight: < 30 kg or > 300 kg (physiologically impossible for adults)
        - Blood pressure ap_hi: < 50 or > 300 mmHg
        - Blood pressure ap_lo: < 30 or > 200 mmHg
        - Diastolic > Systolic (physiologically impossible)
        - Gender: must be 1 or 2
        - Cholesterol: must be 1, 2, or 3
        - Glucose: must be 1, 2, or 3
        - Binary columns (smoke, alco, active, cardio): must be 0 or 1
        """
        print("\nChecking for invalid values...")
        initial_rows = len(df)
        
        # Track invalid values
        invalid_masks = {}
        
        # Height validation
        height_invalid = (df['height'] < 100) | (df['height'] > 250)
        invalid_masks['height'] = height_invalid.sum()
        
        # Weight validation
        weight_invalid = (df['weight'] < 30) | (df['weight'] > 300)
        invalid_masks['weight'] = weight_invalid.sum()
        
        # Blood pressure validation
        ap_hi_invalid = (df['ap_hi'] < 50) | (df['ap_hi'] > 300)
        invalid_masks['ap_hi'] = ap_hi_invalid.sum()
        
        ap_lo_invalid = (df['ap_lo'] < 30) | (df['ap_lo'] > 200)
        invalid_masks['ap_lo'] = ap_lo_invalid.sum()
        
        # Physiological impossibility: diastolic > systolic
        bp_relationship_invalid = df['ap_lo'] > df['ap_hi']
        invalid_masks['bp_relationship'] = bp_relationship_invalid.sum()
        
        # Combine all invalid conditions
        invalid_rows = (
            height_invalid | 
            weight_invalid | 
            ap_hi_invalid | 
            ap_lo_invalid | 
            bp_relationship_invalid
        )
        
        # Validate categorical columns
        gender_invalid = ~df['gender'].isin([1, 2])
        invalid_masks['gender'] = gender_invalid.sum()
        
        cholesterol_invalid = ~df['cholesterol'].isin([1, 2, 3])
        invalid_masks['cholesterol'] = cholesterol_invalid.sum()
        
        gluc_invalid = ~df['gluc'].isin([1, 2, 3])
        invalid_masks['gluc'] = gluc_invalid.sum()
        
        # Validate binary columns
        for col in ['smoke', 'alco', 'active', 'cardio']:
            col_invalid = ~df[col].isin([0, 1])
            invalid_masks[col] = col_invalid.sum()
            invalid_rows |= col_invalid
        
        invalid_rows |= gender_invalid | cholesterol_invalid | gluc_invalid
        
        # Report findings
        print("\nInvalid values found:")
        for key, count in invalid_masks.items():
            if count > 0:
                print(f"  {key}: {count} rows")
        
        self.invalid_values_found = invalid_masks
        
        # Remove invalid rows
        df_clean = df[~invalid_rows].copy()
        self.removed_rows = initial_rows - len(df_clean)
        
        print(f"\nRows removed: {self.removed_rows}")
        print(f"Remaining rows: {len(df_clean)}")
        print(f"Percentage removed: {self.removed_rows / initial_rows * 100:.2f}%")
        
        # Check for NaN and infinity after cleaning
        nan_count = df_clean.isnull().sum().sum()
        inf_count = np.isinf(df_clean.select_dtypes(include=[np.number])).sum().sum()
        
        if nan_count > 0:
            print(f"\nWARNING: {nan_count} NaN values found after cleaning!")
        if inf_count > 0:
            print(f"\nWARNING: {inf_count} infinity values found after cleaning!")
        
        return df_clean
    
    def convert_age_to_years(self, df):
        """Convert age from days to years."""
        print("\nConverting age from days to years...")
        df['age'] = df['age'] / 365.25
        print(f"Age range: {df['age'].min():.1f} - {df['age'].max():.1f} years")
        return df
    
    def remove_duplicates(self, df):
        """
        Drop duplicate patient records.

        The raw dataset stores rows with identical measurements under different
        ids. Left in place, train_test_split scatters copies of the same record
        across both sides, which is train/test contamination.

        Two cases:
        - Exact duplicates (same features AND same label): keep the first.
        - Conflicting duplicates (same features, opposite label): drop every
          copy. They cannot be learned, and keeping one means picking a label
          by coin flip.
        """
        print("\nChecking for duplicate records...")

        # Drop id first: it is the only thing making these rows look distinct.
        if 'id' in df.columns:
            df = df.drop(columns=['id'])
            print("Removed 'id' column")

        before = len(df)
        feature_cols = [c for c in df.columns if c != self.target_column]

        df = df.drop_duplicates()
        self.duplicate_rows = before - len(df)
        print(f"Exact duplicate rows removed: {self.duplicate_rows}")

        label_counts = df.groupby(feature_cols, sort=False)[self.target_column].transform('nunique')
        conflicting = label_counts > 1
        self.conflicting_rows = int(conflicting.sum())
        df = df[~conflicting].copy()
        print(f"Conflicting-label rows removed: {self.conflicting_rows} "
              f"({self.conflicting_rows // 2} contradictory pairs)")

        print(f"Rows after de-duplication: {len(df)}")
        return df

    def prepare_features_and_target(self, df):
        """
        Separate features and target.
        Remove ID column.
        Validate that all features are numerical.
        """
        print("\nPreparing features and target...")
        
        # Remove ID column (not a feature)
        if 'id' in df.columns:
            df = df.drop(columns=['id'])
            print("Removed 'id' column")
        
        # Separate target
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found!")
        
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # Validate target
        unique_target = sorted(y.unique())
        if unique_target != [0, 1]:
            raise ValueError(f"Target must contain only [0, 1], found: {unique_target}")
        
        # Store feature columns
        self.feature_columns = list(X.columns)
        
        # Validate all features are numerical
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            raise ValueError(f"Non-numeric columns found: {non_numeric}")
        
        print(f"Features (X): {X.shape[1]} columns")
        print(f"Target (y): {y.shape[0]} samples")
        print(f"Feature columns: {self.feature_columns}")
        print(f"\nTarget distribution:")
        print(y.value_counts().sort_index())
        print(f"Class balance: {y.value_counts(normalize=True).to_dict()}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2):
        """
        Create stratified train/test split.
        Ensures no data leakage.
        """
        print(f"\nSplitting data (test_size={test_size}, stratified)...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=self.random_state
        )
        
        print(f"Train set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"\nTrain target distribution:")
        print(y_train.value_counts().sort_index())
        print(f"\nTest target distribution:")
        print(y_test.value_counts().sort_index())
        
        return X_train, X_test, y_train, y_test
    
    def identify_scaling_columns(self, X):
        """
        Identify which columns need scaling.
        
        Continuous numerical features: age, height, weight, ap_hi, ap_lo
        Already encoded ordinal: cholesterol (1, 2, 3), gluc (1, 2, 3)
        Binary features: gender (1, 2), smoke, alco, active (0, 1)
        
        We'll scale all continuous features for consistency.
        Ordinal and binary features could be left as-is or scaled.
        For StandardScaler, scaling binary/ordinal is acceptable.
        """
        continuous_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
        ordinal_cols = ['cholesterol', 'gluc']
        binary_cols = ['gender', 'smoke', 'alco', 'active']
        
        # Verify all columns are present
        all_expected = continuous_cols + ordinal_cols + binary_cols
        missing = set(all_expected) - set(X.columns)
        if missing:
            print(f"WARNING: Expected columns missing: {missing}")
        
        # For StandardScaler, we'll scale all features for consistency
        # This is a common practice and won't harm ordinal/binary features
        scaling_cols = list(X.columns)
        
        print(f"\nColumns to scale: {len(scaling_cols)}")
        print(f"  Continuous: {continuous_cols}")
        print(f"  Ordinal: {ordinal_cols}")
        print(f"  Binary: {binary_cols}")
        print("All features will be standardized using StandardScaler.")
        
        return scaling_cols
    
    def scale_features(self, X_train, X_test):
        """
        Fit scaler on training data only, then transform both sets.
        This prevents data leakage.
        """
        print("\nScaling features...")
        
        scaling_cols = self.identify_scaling_columns(X_train)
        
        # Fit on training data ONLY
        print("Fitting StandardScaler on training data...")
        self.scaler.fit(X_train[scaling_cols])
        
        # Transform both sets
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[scaling_cols] = self.scaler.transform(X_train[scaling_cols])
        X_test_scaled[scaling_cols] = self.scaler.transform(X_test[scaling_cols])
        
        print("Scaling complete.")
        print(f"Scaler mean: {self.scaler.mean_[:3]}... (first 3 features)")
        print(f"Scaler scale: {self.scaler.scale_[:3]}... (first 3 features)")
        
        return X_train_scaled, X_test_scaled
    
    def validate_processed_data(self, X_train, X_test, y_train, y_test):
        """
        Comprehensive validation of processed data.
        """
        print("\n" + "=" * 80)
        print("VALIDATION")
        print("=" * 80)
        
        all_valid = True
        
        # Check for NaN
        train_nan = X_train.isnull().sum().sum()
        test_nan = X_test.isnull().sum().sum()
        y_train_nan = y_train.isnull().sum()
        y_test_nan = y_test.isnull().sum()
        
        print(f"\nNaN values:")
        print(f"  X_train: {train_nan}")
        print(f"  X_test: {test_nan}")
        print(f"  y_train: {y_train_nan}")
        print(f"  y_test: {y_test_nan}")
        
        if train_nan > 0 or test_nan > 0 or y_train_nan > 0 or y_test_nan > 0:
            print("  ❌ FAILED: NaN values found!")
            all_valid = False
        else:
            print("  ✓ PASSED")
        
        # Check for infinity
        train_inf = np.isinf(X_train.values).sum()
        test_inf = np.isinf(X_test.values).sum()
        
        print(f"\nInfinity values:")
        print(f"  X_train: {train_inf}")
        print(f"  X_test: {test_inf}")
        
        if train_inf > 0 or test_inf > 0:
            print("  ❌ FAILED: Infinity values found!")
            all_valid = False
        else:
            print("  ✓ PASSED")
        
        # Check target values
        y_train_unique = sorted(y_train.unique())
        y_test_unique = sorted(y_test.unique())
        
        print(f"\nTarget values:")
        print(f"  y_train unique: {y_train_unique}")
        print(f"  y_test unique: {y_test_unique}")
        
        if y_train_unique != [0, 1] or y_test_unique != [0, 1]:
            print("  ❌ FAILED: Invalid target values!")
            all_valid = False
        else:
            print("  ✓ PASSED")
        
        # Check feature columns match
        print(f"\nFeature consistency:")
        print(f"  X_train columns: {list(X_train.columns)}")
        print(f"  X_test columns: {list(X_test.columns)}")
        
        if list(X_train.columns) != list(X_test.columns):
            print("  ❌ FAILED: Train/test columns don't match!")
            all_valid = False
        else:
            print("  ✓ PASSED")
        
        # Check that 'cardio' and 'id' are not in features
        print(f"\n'cardio' in X_train: {'cardio' in X_train.columns}")
        print(f"'id' in X_train: {'id' in X_train.columns}")
        
        if 'cardio' in X_train.columns or 'id' in X_train.columns:
            print("  ❌ FAILED: Target or ID in features!")
            all_valid = False
        else:
            print("  ✓ PASSED")
        
        # Check all features are numerical
        non_numeric_train = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
        non_numeric_test = X_test.select_dtypes(exclude=[np.number]).columns.tolist()
        
        print(f"\nNon-numeric columns:")
        print(f"  X_train: {non_numeric_train if non_numeric_train else 'None'}")
        print(f"  X_test: {non_numeric_test if non_numeric_test else 'None'}")
        
        if non_numeric_train or non_numeric_test:
            print("  ❌ FAILED: Non-numeric features found!")
            all_valid = False
        else:
            print("  ✓ PASSED")
        
        print("\n" + "=" * 80)
        if all_valid:
            print("✓ ALL VALIDATION CHECKS PASSED")
        else:
            print("❌ SOME VALIDATION CHECKS FAILED")
        print("=" * 80)
        
        return all_valid
    
    def save_processed_data(self, X_train, X_test, y_train, y_test, output_dir):
        """Save processed data and scaler."""
        print(f"\nSaving processed data to {output_dir}...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as CSV
        train_df = X_train.copy()
        train_df[self.target_column] = y_train.values
        train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
        
        test_df = X_test.copy()
        test_df[self.target_column] = y_test.values
        test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
        
        # Save scaler
        scaler_path = os.path.join(output_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        
        print(f"Saved:")
        print(f"  - train.csv ({train_df.shape})")
        print(f"  - test.csv ({test_df.shape})")
        print(f"  - scaler.pkl")
        
        return train_df, test_df
    
    def print_summary(self, initial_shape, final_train_shape, final_test_shape):
        """Print comprehensive preprocessing summary."""
        print("\n" + "=" * 80)
        print("PREPROCESSING SUMMARY")
        print("=" * 80)
        
        print(f"\nInitial dataset shape: {initial_shape}")
        print(f"Final train shape: {final_train_shape}")
        print(f"Final test shape: {final_test_shape}")
        print(f"Total final samples: {final_train_shape[0] + final_test_shape[0]}")
        
        print(f"\nRows removed (invalid): {self.removed_rows}")
        print(f"Rows removed (duplicate): {self.duplicate_rows}")
        print(f"Rows removed (conflicting labels): {self.conflicting_rows}")
        total_removed = self.removed_rows + self.duplicate_rows + self.conflicting_rows
        print(f"Total removed: {total_removed} "
              f"({total_removed / initial_shape[0] * 100:.2f}%)")
        
        print(f"\nInvalid values found:")
        for key, count in self.invalid_values_found.items():
            if count > 0:
                print(f"  {key}: {count}")
        
        print(f"\nFeature columns ({len(self.feature_columns)}):")
        for col in self.feature_columns:
            print(f"  - {col}")
        
        print(f"\nTarget column: {self.target_column}")
        
        print(f"\nPreprocessing performed:")
        print(f"  ✓ Invalid value removal")
        print(f"  ✓ Duplicate and conflicting-record removal")
        print(f"  ✓ Age conversion (days → years)")
        print(f"  ✓ ID column removal")
        print(f"  ✓ Feature/target separation")
        print(f"  ✓ Stratified train/test split (80/20)")
        print(f"  ✓ StandardScaler fitting (training data only)")
        print(f"  ✓ Feature scaling (all numerical features)")
        
        print(f"\nCategorical encoding: NONE REQUIRED")
        print(f"  (All categorical features already numerically encoded)")
        
        print(f"\nScaling method: StandardScaler")
        print(f"  (All features scaled to mean=0, std=1)")
        
        print("\n" + "=" * 80)


def main():
    """Run the complete preprocessing pipeline."""
    
    # Configuration
    RAW_DATA_PATH = 'data/raw/cardio_train.csv'
    OUTPUT_DIR = 'data/processed'
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    
    # Initialize preprocessor
    preprocessor = CardioPreprocessor(random_state=RANDOM_STATE)
    
    # Load data
    df = preprocessor.load_data(RAW_DATA_PATH)
    initial_shape = df.shape
    
    # Handle invalid values
    df_clean = preprocessor.handle_invalid_values(df)
    
    # Convert age to years
    df_clean = preprocessor.convert_age_to_years(df_clean)
    
    # Remove duplicate records (prevents train/test contamination)
    df_clean = preprocessor.remove_duplicates(df_clean)

    # Prepare features and target
    X, y = preprocessor.prepare_features_and_target(df_clean)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, test_size=TEST_SIZE)
    
    # Scale features
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    # Validate
    validation_passed = preprocessor.validate_processed_data(
        X_train_scaled, X_test_scaled, y_train, y_test
    )
    
    if not validation_passed:
        print("\n❌ Validation failed! Please review the issues above.")
        return
    
    # Save processed data
    train_df, test_df = preprocessor.save_processed_data(
        X_train_scaled, X_test_scaled, y_train, y_test, OUTPUT_DIR
    )
    
    # Print summary
    preprocessor.print_summary(
        initial_shape,
        X_train_scaled.shape,
        X_test_scaled.shape
    )
    
    # Verify saved data by loading it back
    print("\nVerifying saved data...")
    train_reloaded = pd.read_csv(os.path.join(OUTPUT_DIR, 'train.csv'))
    test_reloaded = pd.read_csv(os.path.join(OUTPUT_DIR, 'test.csv'))
    scaler_reloaded = joblib.load(os.path.join(OUTPUT_DIR, 'scaler.pkl'))
    
    print(f"✓ Train data reloaded: {train_reloaded.shape}")
    print(f"✓ Test data reloaded: {test_reloaded.shape}")
    print(f"✓ Scaler reloaded: {type(scaler_reloaded).__name__}")
    
    print("\n✓ PREPROCESSING COMPLETE!")


if __name__ == '__main__':
    main()
