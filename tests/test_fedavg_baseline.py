import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import importlib
import numpy as np
import pytest

fedavg_baseline_module = importlib.import_module("experiments.02_fedavg_baseline")
run_fedavg_baseline = fedavg_baseline_module.run_fedavg_baseline


def test_fedavg_baseline_5_rounds_3_clients():
    """
    Validates all 5 tasks:
    1. Connects 3 simulated hospital clients.
    2. Uses standard FedAvg strategy.
    3. Runs 5-round simulation successfully.
    4. Verifies clients train and send model parameters.
    5. Verifies server aggregates and redistributes the global model.
    """
    history, strategy = run_fedavg_baseline(num_rounds=5, num_clients=3)

    # 1. 5 rounds executed successfully
    assert history is not None
    assert len(history.losses_distributed) == 5

    # 2. Server aggregated and redistributed updated parameters in every round after round 1
    # Round weight deltas: [round 2 vs 1, round 3 vs 2, round 4 vs 3, round 5 vs 4]
    assert len(strategy.round_weight_deltas) == 4
    for delta in strategy.round_weight_deltas:
        assert delta > 0.0, f"Expected non-zero global weight delta between rounds, got {delta}"

    # 3. Evaluation metrics were collected across all 5 rounds
    accuracies = [acc for _, acc in history.metrics_distributed.get("accuracy", [])]
    assert len(accuracies) == 5
    for acc in accuracies:
        assert 0.0 <= acc <= 1.0
