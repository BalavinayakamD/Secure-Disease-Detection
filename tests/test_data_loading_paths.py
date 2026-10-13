"""
Test data loading path resolution.

Ensures that preprocessing functions resolve paths correctly relative to project root.
"""

import os
import sys
from pathlib import Path
import pytest

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.preprocessing import _get_project_root, load_processed_data


def test_get_project_root():
    """Test that project root is correctly identified."""
    root = _get_project_root()
    assert root.exists()
    assert (root / 'config.yaml').exists()
    assert (root / 'src').exists()
    assert (root / 'data').exists()


def test_load_processed_data_with_relative_path():
    """Test that load_processed_data resolves relative paths correctly."""
    # This should raise FileNotFoundError with clear message if files don't exist
    with pytest.raises(FileNotFoundError) as exc_info:
        load_processed_data('data/processed')
    
    assert 'train.csv' in str(exc_info.value)
    assert 'data/processed' in str(exc_info.value).lower()


def test_load_processed_data_with_absolute_path():
    """Test that load_processed_data handles absolute paths."""
    abs_path = Path(__file__).resolve().parents[1] / 'data' / 'processed'
    
    with pytest.raises(FileNotFoundError) as exc_info:
        load_processed_data(str(abs_path))
    
    assert 'train.csv' in str(exc_info.value)


def test_missing_data_error_message():
    """Test that missing data produces actionable error message."""
    try:
        load_processed_data('data/processed')
        pytest.fail("Should have raised FileNotFoundError")
    except FileNotFoundError as e:
        error_msg = str(e)
        assert 'train.csv' in error_msg or 'test.csv' in error_msg
        assert 'preprocessing' in error_msg.lower() or 'hospital' in error_msg.lower()


def test_hospital_data_exists():
    """Check if hospital-specific data files exist."""
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / 'data' / 'processed'
    
    # Check for hospital_0_train.csv as a smoke test
    hospital_0_train = data_dir / 'hospital_0_train.csv'
    
    # This test documents the current state
    # If files exist, great. If not, we know preprocessing is needed.
    exists = hospital_0_train.exists()
    
    if not exists:
        pytest.skip(
            f"Hospital data not found at {hospital_0_train}. "
            "Run data preprocessing to generate hospital splits."
        )
    else:
        # If files exist, verify they have expected structure
        import pandas as pd
        df = pd.read_csv(hospital_0_train)
        assert 'cardio' in df.columns, "Hospital data missing 'cardio' column"
        assert len(df) > 0, "Hospital data is empty"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
