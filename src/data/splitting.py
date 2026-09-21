"""
Dirichlet Non-IID splitting of the cardio training set into simulated hospitals.

Each class is partitioned independently with proportions drawn from a Dirichlet,
which is what produces label skew (not merely size skew) across hospitals.
"""

import os

import numpy as np
import pandas as pd
import yaml


def dirichlet_split(
    df,
    num_clients=3,
    alpha=1.0,
    min_per_class=20,
    seed=42,
    target='cardio',
):
    """
    Split a labelled DataFrame into `num_clients` Non-IID shards.

    Args:
        df: DataFrame containing features plus the `target` column.
        num_clients: Number of simulated hospitals.
        alpha: Dirichlet concentration. Lower = more skew (0.1 severe, 100 ~ IID).
        min_per_class: Every hospital must hold at least this many rows of every class.
        seed: Base seed. Redraws use seed + attempt, so the result is reproducible.
        target: Name of the label column.

    Returns:
        list[DataFrame]: One shard per hospital, rows shuffled.

    Raises:
        ValueError: If the min_per_class constraint cannot be met in 100 attempts.
    """
    classes = sorted(df[target].unique())

    for attempt in range(100):
        rng = np.random.default_rng(seed + attempt)
        # parts[i] collects the index arrays handed to hospital i, one per class.
        parts = [[] for _ in range(num_clients)]
        satisfied = True

        for c in classes:
            # permutation, not shuffle: pandas hands back read-only arrays.
            idx = rng.permutation(df.index[df[target] == c].to_numpy())

            proportions = rng.dirichlet(np.full(num_clients, alpha))
            cuts = (np.cumsum(proportions)[:-1] * len(idx)).astype(int)

            for i, chunk in enumerate(np.split(idx, cuts)):
                if len(chunk) < min_per_class:
                    satisfied = False
                parts[i].append(chunk)

        if satisfied:
            # NOTE: redraw on violation instead of the architecture doc's forced
            # inter-hospital swap. At alpha=1.0 with ~27k rows per class it never
            # fires. Measured floor: alpha >= 0.1 is reachable by redrawing,
            # alpha = 0.05 is not -- implement the swap if you need that regime.
            shards = []
            for p in parts:
                merged = rng.permutation(np.concatenate(p))
                shards.append(df.loc[merged].reset_index(drop=True))
            return shards

    raise ValueError(
        f'Could not satisfy min_per_class={min_per_class} for {num_clients} clients '
        f'at alpha={alpha} in 100 attempts. Raise alpha or lower min_per_class.'
    )


def class_counts(shards, target='cardio'):
    """Per-hospital class counts as a DataFrame, for logging and sanity checks."""
    return pd.DataFrame(
        [s[target].value_counts().sort_index().to_dict() for s in shards],
        index=[f'hospital_{i}' for i in range(len(shards))],
    ).fillna(0).astype(int)


def main():
    with open('config.yaml') as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg['data']
    processed_dir = data_cfg['processed_dir']

    train = pd.read_csv(os.path.join(processed_dir, 'train.csv'))
    print(f'Loaded train.csv: {train.shape}')

    shards = dirichlet_split(
        train,
        num_clients=data_cfg['num_clients'],
        alpha=data_cfg['dirichlet_alpha'],
        min_per_class=data_cfg['min_samples_per_class'],
        seed=data_cfg['random_seed'],
    )

    counts = class_counts(shards)
    counts['total'] = counts.sum(axis=1)
    counts['pos_rate'] = (counts[1] / counts['total']).round(4)

    print(f"\nDirichlet alpha={data_cfg['dirichlet_alpha']}, "
          f"min_per_class={data_cfg['min_samples_per_class']}")
    print(counts.to_string())

    for i, shard in enumerate(shards):
        path = os.path.join(processed_dir, f'hospital_{i}.csv')
        shard.to_csv(path, index=False)
        print(f'Saved {path} ({shard.shape})')

    print('\nTest set (data/processed/test.csv) is untouched and held out.')


if __name__ == '__main__':
    main()
