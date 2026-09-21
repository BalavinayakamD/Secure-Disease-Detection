"""Checks for the Dirichlet Non-IID splitter."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.splitting import class_counts, dirichlet_split, local_train_test_split


def make_df(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'f1': rng.normal(size=n),
        'f2': rng.normal(size=n),
        'cardio': rng.integers(0, 2, size=n),
    })


def test_partition_is_exact():
    """Shards must tile the input: no loss, no duplication."""
    df = make_df()
    shards = dirichlet_split(df, num_clients=3, alpha=1.0, min_per_class=20)

    assert sum(len(s) for s in shards) == len(df)

    rebuilt = pd.concat(shards).sort_values(['f1', 'f2']).reset_index(drop=True)
    original = df.sort_values(['f1', 'f2']).reset_index(drop=True)
    pd.testing.assert_frame_equal(rebuilt, original)


def test_min_per_class_constraint():
    df = make_df()
    counts = class_counts(dirichlet_split(df, num_clients=3, alpha=1.0, min_per_class=20))

    assert set(counts.columns) == {0, 1}, 'every hospital must hold both classes'
    assert (counts >= 20).all().all()


def test_seed_is_reproducible():
    df = make_df()
    a = dirichlet_split(df, seed=42)
    b = dirichlet_split(df, seed=42)
    for sa, sb in zip(a, b):
        pd.testing.assert_frame_equal(sa, sb)

    c = dirichlet_split(df, seed=7)
    assert not all(len(sa) == len(sc) for sa, sc in zip(a, c))


def test_low_alpha_is_more_skewed():
    """The whole point of alpha: small alpha => label distributions diverge."""
    df = make_df(n=20000)

    def mean_spread(alpha):
        # Average over seeds: a single Dirichlet draw is far too noisy to compare.
        spreads = []
        for seed in range(5):
            counts = class_counts(
                dirichlet_split(df, alpha=alpha, min_per_class=5, seed=seed)
            )
            pos_rate = counts[1] / counts.sum(axis=1)
            spreads.append(pos_rate.max() - pos_rate.min())
        return sum(spreads) / len(spreads)

    assert mean_spread(0.3) > 3 * mean_spread(100.0)


def test_impossible_constraint_raises():
    df = make_df(n=60)
    with pytest.raises(ValueError, match='min_per_class'):
        dirichlet_split(df, num_clients=3, alpha=1.0, min_per_class=500)


def test_local_split_preserves_hospital_skew():
    """Each hospital's local test set must look like that hospital, not like the pool."""
    df = make_df(n=20000)
    # alpha=0.3 to guarantee the hospitals actually differ in label rate.
    shards = dirichlet_split(df, alpha=0.3, min_per_class=20, seed=3)
    splits = local_train_test_split(shards, test_frac=0.15, seed=42)

    assert len(splits) == len(shards)
    for shard, (tr, te) in zip(shards, splits):
        assert len(tr) + len(te) == len(shard), 'local split must not lose rows'
        # Stratified: the local test set carries the shard's own positive rate.
        assert abs(te['cardio'].mean() - shard['cardio'].mean()) < 0.02


def test_local_split_is_disjoint():
    df = make_df(n=3000)
    shards = dirichlet_split(df, alpha=1.0, min_per_class=20, seed=1)
    for shard, (tr, te) in zip(shards, local_train_test_split(shards)):
        merged = pd.concat([tr, te]).sort_values(['f1', 'f2']).reset_index(drop=True)
        expected = shard.sort_values(['f1', 'f2']).reset_index(drop=True)
        pd.testing.assert_frame_equal(merged, expected)


def test_majority_baseline_flags_degenerate_model():
    """A constant predictor must score high accuracy but 0.5 balanced accuracy."""
    from src.utils.metrics import binary_metrics

    y_true = np.array([1] * 83 + [0] * 17)
    y_prob = np.full(100, 0.99)  # always says "positive"
    m = binary_metrics(y_true, y_prob)

    assert m['accuracy'] == pytest.approx(0.83)
    assert m['majority_baseline'] == pytest.approx(0.83)
    assert m['balanced_accuracy'] == pytest.approx(0.5)
