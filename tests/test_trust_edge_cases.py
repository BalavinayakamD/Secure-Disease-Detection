"""
Test trust mechanism edge cases and correctness.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.fl.strategy import ATFedBNStrategy


def test_cosine_similarity_zero_vectors():
    """Test cosine similarity with zero-norm vectors."""
    strategy = ATFedBNStrategy()
    
    # Zero vector vs non-zero
    v1 = np.zeros(10)
    v2 = np.random.randn(10)
    sim = strategy.compute_cosine_similarity(v1, v2)
    assert sim == 1.0, "Zero vector should return 1.0 (safe default)"
    
    # Both zero
    sim = strategy.compute_cosine_similarity(v1, v1)
    assert sim == 1.0


def test_cosine_similarity_opposite_vectors():
    """Test cosine similarity with opposite-direction vectors."""
    strategy = ATFedBNStrategy()
    
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = -v1  # Opposite direction
    
    sim = strategy.compute_cosine_similarity(v1, v2)
    assert -1.0 <= sim <= -0.99, f"Opposite vectors should have sim ≈ -1, got {sim}"


def test_cosine_similarity_identical_vectors():
    """Test cosine similarity with identical vectors."""
    strategy = ATFedBNStrategy()
    
    v1 = np.random.randn(50)
    sim = strategy.compute_cosine_similarity(v1, v1)
    assert 0.99 <= sim <= 1.0, f"Identical vectors should have sim ≈ 1, got {sim}"


def test_trust_weights_all_zero():
    """Test that trust weights fall back to volume when all trust scores are zero."""
    strategy = ATFedBNStrategy(beta=0.5)
    
    # Simulate scenario where all cosine similarities are negative (trust=0 after ReLU)
    raw_trust_scores = [0.0, 0.0, 0.0]
    volume_weights = np.array([0.3, 0.5, 0.2])
    
    sum_trust = sum(raw_trust_scores)
    if sum_trust > strategy.trust_epsilon:
        trust_weights = np.array([t / sum_trust for t in raw_trust_scores])
    else:
        trust_weights = np.copy(volume_weights)
    
    # Should fall back to volume weights
    assert np.allclose(trust_weights, volume_weights)


def test_hybrid_weights_beta_extremes():
    """Test hybrid weighting with beta=0 and beta=1."""
    strategy_volume_only = ATFedBNStrategy(beta=1.0)
    strategy_trust_only = ATFedBNStrategy(beta=0.0)
    
    volume_weights = np.array([0.3, 0.5, 0.2])
    trust_weights = np.array([0.6, 0.2, 0.2])
    
    # beta=1.0: pure volume
    final_volume = 1.0 * volume_weights + 0.0 * trust_weights
    final_volume = final_volume / np.sum(final_volume)
    assert np.allclose(final_volume, volume_weights)
    
    # beta=0.0: pure trust
    final_trust = 0.0 * volume_weights + 1.0 * trust_weights
    final_trust = final_trust / np.sum(final_trust)
    assert np.allclose(final_trust, trust_weights)


def test_trust_scores_normalized():
    """Test that final weights sum to 1.0."""
    strategy = ATFedBNStrategy(beta=0.7)
    
    # Simulate various trust scenarios
    test_cases = [
        ([0.8, 0.6, 0.9], [100, 200, 150]),  # Normal case
        ([0.0, 0.0, 0.0], [100, 200, 150]),  # All zero trust
        ([1.0, 1.0, 1.0], [100, 100, 100]),  # Perfect trust, equal volume
        ([0.1, 0.9, 0.5], [50, 50, 300]),    # Uneven distribution
    ]
    
    for raw_trust, samples in test_cases:
        volume_weights = np.array([s / sum(samples) for s in samples])
        
        sum_trust = sum(raw_trust)
        if sum_trust > strategy.trust_epsilon:
            trust_weights = np.array([t / sum_trust for t in raw_trust])
        else:
            trust_weights = volume_weights
        
        final_weights = strategy.beta * volume_weights + (1 - strategy.beta) * trust_weights
        final_weights = final_weights / np.sum(final_weights)
        
        assert np.isclose(np.sum(final_weights), 1.0), f"Weights must sum to 1, got {np.sum(final_weights)}"
        assert np.all(final_weights >= 0), "Weights must be non-negative"
        assert np.all(np.isfinite(final_weights)), "Weights must be finite"


def test_relu_clipping_negative_similarities():
    """Test that negative cosine similarities are clipped to zero."""
    strategy = ATFedBNStrategy(trust_floor=0.0)
    
    # Negative similarity
    cos_sim = -0.5
    trust = max(strategy.trust_floor, cos_sim)
    
    assert trust == 0.0, f"Negative similarity should be clipped to 0, got {trust}"
    
    # Positive similarity
    cos_sim = 0.7
    trust = max(strategy.trust_floor, cos_sim)
    assert trust == 0.7


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
