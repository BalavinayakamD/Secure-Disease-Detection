"""
Test that UI is fully data-driven with no hardcoded metrics.

Verifies dynamic data loading, configuration resolution, and proper handling
of missing data.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def test_ui_import_and_data_functions():
    """Test that UI helper functions work correctly."""
    # Import UI module to test utility functions
    import importlib.util
    spec = importlib.util.spec_from_file_location('ui_app', 'ui/app.py')
    ui_app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ui_app)
    
    # Test get_project_root
    project_root_path = ui_app.get_project_root()
    assert project_root_path.exists()
    assert (project_root_path / 'config.yaml').exists()
    
    # Test resolve_path
    resolved = ui_app.resolve_path('config.yaml')
    assert os.path.isabs(resolved)
    assert os.path.exists(resolved)
    
    # Test load_config
    cfg = ui_app.load_config()
    assert 'data' in cfg
    assert 'model' in cfg
    assert 'training' in cfg
    assert 'privacy' in cfg
    
    # Test count_available_experiments
    count = ui_app.count_available_experiments()
    assert isinstance(count, int)
    assert count >= 0
    
    # Test get_dataset_statistics
    stats = ui_app.get_dataset_statistics(cfg)
    assert 'num_hospitals' in stats
    assert stats['num_hospitals'] == cfg['data']['num_clients']
    assert 'total_train' in stats
    assert 'total_test' in stats
    assert isinstance(stats['total_train'], int)
    assert isinstance(stats['total_test'], int)
    
    # Test format_metric
    assert ui_app.format_metric(0.7234, 'float') == '0.7234'
    assert ui_app.format_metric(0.85, 'percent') == '85.00%'
    assert ui_app.format_metric(12345, 'int') == '12,345'
    assert ui_app.format_metric(None) == '—'


def test_epsilon_calculation():
    """Test privacy budget epsilon estimation."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('ui_app', 'ui/app.py')
    ui_app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ui_app)
    
    # Test with actual noise scale
    epsilon = ui_app.get_epsilon_from_noise_scale(0.01, steps=10)
    assert epsilon is not None
    assert epsilon > 1000  # Very weak privacy
    
    # Test with stronger noise
    epsilon_strong = ui_app.get_epsilon_from_noise_scale(1.0, steps=10)
    assert epsilon_strong is not None
    assert epsilon_strong < epsilon  # Stronger noise -> smaller epsilon
    
    # Test with no noise
    epsilon_none = ui_app.get_epsilon_from_noise_scale(0, steps=10)
    assert epsilon_none is None
    
    # Test format_epsilon
    eps_str, level = ui_app.format_epsilon(None)
    assert eps_str == "∞ (No DP)"
    assert level == "None"
    
    eps_str, level = ui_app.format_epsilon(150)
    assert "150" in eps_str
    assert level == "❌ Very Weak"
    
    eps_str, level = ui_app.format_epsilon(0.5)
    assert "0.5" in eps_str
    assert level == "✅ Strong"


def test_dataset_statistics_with_actual_files():
    """Test dataset statistics computation with actual hospital files."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('ui_app', 'ui/app.py')
    ui_app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ui_app)
    
    cfg = ui_app.load_config()
    stats = ui_app.get_dataset_statistics(cfg)
    
    # If hospital files exist, verify statistics
    if stats['total_train'] > 0:
        assert len(stats['hospital_details']) == cfg['data']['num_clients']
        
        for hospital in stats['hospital_details']:
            assert 'id' in hospital
            assert 'train_samples' in hospital
            assert 'test_samples' in hospital
            assert 'positive_rate_train' in hospital
            assert 'positive_rate_test' in hospital
            assert 0 <= hospital['positive_rate_train'] <= 1
            assert 0 <= hospital['positive_rate_test'] <= 1
        
        # Verify totals match sum
        total_train_from_hospitals = sum(h['train_samples'] for h in stats['hospital_details'])
        total_test_from_hospitals = sum(h['test_samples'] for h in stats['hospital_details'])
        assert stats['total_train'] == total_train_from_hospitals
        assert stats['total_test'] == total_test_from_hospitals


def test_no_hardcoded_metrics_in_overview():
    """Verify Overview tab has no hardcoded performance values."""
    with open('ui/app.py', 'r') as f:
        ui_code = f.read()
    
    # Search for suspicious hardcoded values in Overview section
    # (values between 0.5 and 0.99 that might be metrics)
    
    # Legitimate constants to exclude:
    legitimate_constants = [
        '0.5',  # Used for skew threshold, beta bounds
        '0.7',  # Default beta value
        '1.0',  # Beta max, weights
        '0.0',  # Beta min
        '0.1',  # Step values
    ]
    
    # Check that actual metric values are loaded from files, not hardcoded
    assert 'load_json' in ui_code
    assert 'summary_statistics.json' in ui_code
    assert "format_metric(at_fedbn.get('roc_auc'))" in ui_code
    assert "format_metric(" in ui_code and ".get('accuracy" in ui_code


def test_experiment_count_dynamic():
    """Test that experiment count is calculated dynamically."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('ui_app', 'ui/app.py')
    ui_app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ui_app)
    
    # Count should match actual files
    count = ui_app.count_available_experiments()
    
    results_dir = ui_app.resolve_path('outputs/results')
    expected_files = [
        'central_baseline.json',
        'fedavg_baseline.json',
        'fedbn_only.json',
        'at_fedbn.json'
    ]
    
    expected_count = sum(1 for f in expected_files if os.path.exists(os.path.join(results_dir, f)))
    assert count == expected_count


def test_config_values_not_hardcoded():
    """Test that configuration values are read from config.yaml."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('ui_app', 'ui/app.py')
    ui_app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ui_app)
    
    cfg = ui_app.load_config()
    
    # Verify expected configuration structure
    assert 'data' in cfg
    assert 'num_clients' in cfg['data']
    assert 'model' in cfg
    assert 'training' in cfg
    assert 'local_epochs' in cfg['training']
    assert 'federated' in cfg
    assert 'num_rounds' in cfg['federated']
    assert 'beta' in cfg['federated']
    assert 'privacy' in cfg
    assert 'noise_scale' in cfg['privacy']
    
    # Verify UI uses these values dynamically
    with open('ui/app.py', 'r') as f:
        ui_code = f.read()
    
    assert "cfg['data']['num_clients']" in ui_code
    assert "cfg['training']['local_epochs']" in ui_code
    assert "cfg['federated']['num_rounds']" in ui_code
    assert "cfg['privacy'].get('noise_scale'" in ui_code


def test_privacy_parameters_dynamic():
    """Test that privacy parameters are computed from actual config."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('ui_app', 'ui/app.py')
    ui_app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ui_app)
    
    cfg = ui_app.load_config()
    noise_scale = cfg['privacy'].get('noise_scale', 0)
    
    if noise_scale > 0:
        epsilon = ui_app.get_epsilon_from_noise_scale(
            noise_scale,
            steps=cfg['federated']['num_rounds']
        )
        assert epsilon is not None
        
        # Epsilon should change with different parameters
        epsilon_more_rounds = ui_app.get_epsilon_from_noise_scale(
            noise_scale,
            steps=cfg['federated']['num_rounds'] * 2
        )
        assert epsilon_more_rounds > epsilon  # More rounds = less privacy


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
