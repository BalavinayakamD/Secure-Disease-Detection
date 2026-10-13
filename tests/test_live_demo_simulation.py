"""
Test Live Demo simulation integration.

Ensures the UI can invoke the AT-FedBN experiment without crashes.
"""

import sys
from pathlib import Path
import pytest

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)


@pytest.mark.slow
def test_live_demo_smoke():
    """Smoke test: Run 1-round AT-FedBN simulation."""
    try:
        from experiments.experiment_04_full_at_fedbn import run_at_fedbn
    except ImportError:
        # Module name has dots, try direct import
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'exp04', 'experiments/04_full_at_fedbn.py'
        )
        exp04 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exp04)
        run_at_fedbn = exp04.run_at_fedbn
    
    # Run 1-round simulation
    strategy, results, payload = run_at_fedbn(
        num_rounds=1,
        num_clients=3,
        beta=0.7,
        tag='pytest_smoke'
    )
    
    # Verify results structure
    assert results is not None
    assert 'experiment' in results
    assert results['experiment'] == 'at_fedbn'
    assert 'num_rounds' in results
    assert 'local' in results
    assert 'per_hospital' in results['local']
    assert 'average' in results['local']
    
    # Verify payload structure
    assert payload is not None
    assert 'trust_scores' in payload
    assert 'final_metrics' in payload
    assert 'hospital_metrics' in payload
    
    # Verify we got results for all hospitals
    assert len(results['local']['per_hospital']) == 3
    
    print("\n✓ Live Demo simulation smoke test passed")


def test_hospital_data_loading():
    """Test that hospital data can be loaded."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'exp04', 'experiments/04_full_at_fedbn.py'
    )
    exp04 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp04)
    
    # Test loading hospital 0
    train_dataset, test_dataset = exp04.load_hospital_data(0)
    
    assert len(train_dataset) > 0
    assert len(test_dataset) > 0
    
    # Check data structure
    X_sample, y_sample = train_dataset[0]
    assert X_sample.shape[0] == 11  # 11 features
    assert y_sample.shape == ()  # Scalar label


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
