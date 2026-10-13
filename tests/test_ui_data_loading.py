"""
Test UI data loading functions.

Ensures the dashboard can load existing results without errors.
"""

import json
import os
import sys
from pathlib import Path

import pytest
import yaml

# Add project root to path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def test_config_loading():
    """Test that config.yaml loads successfully."""
    with open('config.yaml') as f:
        cfg = yaml.safe_load(f)
    
    assert 'data' in cfg
    assert 'model' in cfg
    assert 'training' in cfg
    assert 'privacy' in cfg
    
    # Check privacy keys exist (UI depends on these)
    assert 'noise_scale' in cfg['privacy']
    assert 'clip_norm' in cfg['privacy']


def test_load_summary_statistics():
    """Test loading summary_statistics.json with correct keys."""
    path = 'outputs/results/summary_statistics.json'
    assert os.path.exists(path), "summary_statistics.json not found"
    
    with open(path) as f:
        summary = json.load(f)
    
    # Keys that UI expects
    assert 'experiments_completed' in summary
    assert 'num_hospitals' in summary
    assert 'num_fl_rounds' in summary
    assert 'total_training_samples' in summary
    assert 'total_test_samples' in summary
    assert 'performance_comparison' in summary
    
    # Check performance comparison structure
    perf = summary['performance_comparison']
    assert 'Centralized' in perf
    assert 'AT-FedBN' in perf
    
    # Check each has required metrics
    for strategy in perf.values():
        assert 'accuracy' in strategy
        assert 'balanced_accuracy' in strategy
        assert 'f1' in strategy
        assert 'roc_auc' in strategy


def test_load_at_fedbn_results():
    """Test loading at_fedbn.json with correct structure."""
    path = 'outputs/results/at_fedbn.json'
    assert os.path.exists(path), "at_fedbn.json not found"
    
    with open(path) as f:
        data = json.load(f)
    
    assert 'experiment' in data
    assert 'num_rounds' in data
    assert 'beta' in data
    assert 'local' in data
    
    # Check local results structure
    local = data['local']
    assert 'per_hospital' in local
    assert 'average' in local
    
    # Check average has required metrics
    avg = local['average']
    assert 'accuracy' in avg
    assert 'balanced_accuracy' in avg
    assert 'f1' in avg
    assert 'roc_auc' in avg


def test_load_fedbn_results():
    """Test loading fedbn_only.json with correct structure."""
    path = 'outputs/results/fedbn_only.json'
    assert os.path.exists(path), "fedbn_only.json not found"
    
    with open(path) as f:
        data = json.load(f)
    
    assert 'experiment' in data
    assert 'num_rounds' in data
    assert 'local' in data
    
    # Check local results structure
    local = data['local']
    assert 'per_hospital' in local
    assert 'average' in local
    
    # Check average has required metrics
    avg = local['average']
    assert 'accuracy' in avg
    assert 'balanced_accuracy' in avg
    assert 'f1' in avg
    assert 'roc_auc' in avg


def test_load_central_baseline_results():
    """Test loading central_baseline.json with correct structure."""
    path = 'outputs/results/central_baseline.json'
    assert os.path.exists(path), "central_baseline.json not found"
    
    with open(path) as f:
        data = json.load(f)
    
    assert 'experiment' in data
    assert 'best' in data
    
    # Check best has required metrics
    best = data['best']
    assert 'accuracy' in best
    assert 'balanced_accuracy' in best
    assert 'f1' in best
    assert 'roc_auc' in best


def test_load_sigma_sweep_results():
    """Test loading sigma_sweep.json with correct structure."""
    path = 'outputs/results/sigma_sweep.json'
    assert os.path.exists(path), "sigma_sweep.json not found"
    
    with open(path) as f:
        data = json.load(f)
    
    assert 'sweep_type' in data
    assert 'sweep' in data
    
    # Check sweep entries
    for entry in data['sweep']:
        assert 'sigma' in entry
        assert 'epsilon' in entry or entry['epsilon'] is None
        assert 'local' in entry
        
        # Check local metrics
        local = entry['local']
        assert 'accuracy' in local
        assert 'roc_auc' in local


def test_master_table_exists():
    """Test that master_table.md exists."""
    path = 'outputs/results/master_table.md'
    assert os.path.exists(path), "master_table.md not found"
    
    with open(path) as f:
        content = f.read()
    
    # Check it has meaningful content
    assert 'Centralized' in content
    assert 'FedBN' in content
    assert 'AT-FedBN' in content
    assert 'ROC-AUC' in content


def test_ui_import_without_errors():
    """Test that ui/app.py can be imported without errors."""
    # This test ensures the UI code is syntactically correct
    # and doesn't fail on import
    try:
        import ui.app
        assert True
    except Exception as e:
        pytest.fail(f"UI app failed to import: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
