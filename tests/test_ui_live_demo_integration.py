"""
Test UI Live Demo button integration.

Simulates the exact code path triggered when user clicks "Run 2-Round Training" button.
"""

import sys
from pathlib import Path
import pytest

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def test_ui_live_demo_button_click_simulation():
    """
    Simulate the exact UI Live Demo button click.
    
    This matches the code in ui/app.py tab_demo section:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'exp04', 'experiments/04_full_at_fedbn.py'
        )
        exp04 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exp04)
        
        strategy, results, payload = exp04.run_at_fedbn(
            num_rounds=2,
            beta=beta_demo,
            tag='ui_live_demo'
        )
    """
    import importlib.util
    
    # Exact import pattern from UI
    spec = importlib.util.spec_from_file_location(
        'exp04', 'experiments/04_full_at_fedbn.py'
    )
    exp04 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp04)
    
    # Run with UI parameters
    beta_demo = 0.7
    strategy, results, payload = exp04.run_at_fedbn(
        num_rounds=2,
        beta=beta_demo,
        tag='ui_live_demo_test'
    )
    
    # Verify the outputs UI expects
    assert strategy is not None, "Strategy should not be None"
    assert results is not None, "Results should not be None"
    assert payload is not None, "Payload should not be None"
    
    # Check results structure (used in UI trust scores table)
    assert 'local' in results
    assert 'per_hospital' in results['local']
    
    # Check payload structure (used in UI display)
    assert 'trust_scores' in payload
    assert 'hospital_metrics' in payload
    
    # Verify trust scores per round (UI displays these in dataframe)
    trust_scores = payload['trust_scores']
    assert len(trust_scores) == 2, "Should have trust scores for 2 rounds"
    
    for round_num, scores in trust_scores.items():
        assert isinstance(scores, dict), "Each round should have dict of scores"
        assert len(scores) > 0, "Each round should have at least one client score"
        for client_id, score in scores.items():
            assert isinstance(score, float), "Trust scores should be floats"
            assert 0.0 <= score <= 1.0, f"Trust score {score} should be in [0,1]"
    
    # Verify hospital performance (UI displays these in dataframe)
    hospital_metrics = results['local']['per_hospital']
    assert len(hospital_metrics) == 3, "Should have metrics for 3 hospitals"
    
    for hospital, metrics in hospital_metrics.items():
        assert 'roc_auc' in metrics, f"{hospital} missing roc_auc"
        assert 'balanced_accuracy' in metrics, f"{hospital} missing balanced_accuracy"
        assert 'accuracy' in metrics, f"{hospital} missing accuracy"
        assert 'majority_baseline' in metrics, f"{hospital} missing majority_baseline"
        
        # Sanity check values
        assert 0.0 <= metrics['roc_auc'] <= 1.0
        assert 0.0 <= metrics['accuracy'] <= 1.0
    
    print("\n✓ UI Live Demo integration test passed")
    print(f"  Trust scores computed for {len(trust_scores)} rounds")
    print(f"  Hospital metrics collected for {len(hospital_metrics)} hospitals")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
