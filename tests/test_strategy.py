import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pytest
import torch

from flwr.common import (
    FitRes,
    Status,
    Code,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

from src.fl.strategy import ATFedBNStrategy
from src.fl.server import run_simulation


class MockClientProxy(ClientProxy):
    """Mock ClientProxy for unit testing."""
    def __init__(self, cid: str):
        super().__init__(cid)

    def get_properties(self, ins, timeout, group_id=None):
        return None

    def get_parameters(self, ins, timeout, group_id=None):
        return None

    def fit(self, ins, timeout, group_id=None):
        return None

    def evaluate(self, ins, timeout, group_id=None):
        return None

    def reconnect(self, ins, timeout, group_id=None):
        return None


def test_cosine_similarity_computation():
    strategy = ATFedBNStrategy(beta=0.7)

    # Identical vectors
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([1.0, 2.0, 3.0])
    assert pytest.approx(strategy.compute_cosine_similarity(v1, v2), 1e-4) == 1.0

    # Opposite vectors
    v3 = np.array([-1.0, -2.0, -3.0])
    assert pytest.approx(strategy.compute_cosine_similarity(v1, v3), 1e-4) == -1.0

    # Orthogonal vectors
    v4 = np.array([1.0, 0.0, 0.0])
    v5 = np.array([0.0, 1.0, 0.0])
    assert pytest.approx(strategy.compute_cosine_similarity(v4, v5), 1e-4) == 0.0


def test_at_fedbn_trust_and_hybrid_aggregation():
    # Setup initial global parameters (1 layer for simplicity: 2x2 matrix)
    w_init = [np.zeros((2, 2), dtype=np.float32)]
    initial_params = ndarrays_to_parameters(w_init)

    strategy = ATFedBNStrategy(
        beta=0.7,
        trust_floor=0.0,
        initial_parameters=initial_params,
    )

    # Client 1: Good update aligned with global direction (+1.0)
    # 100 samples
    w_client1 = [np.ones((2, 2), dtype=np.float32)]
    res1 = FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=ndarrays_to_parameters(w_client1),
        num_examples=100,
        metrics={"loss": 0.5},
    )

    # Client 2: Poisoned / inverted update (-1.0)
    # 100 samples
    w_client2 = [-np.ones((2, 2), dtype=np.float32) * 0.5]
    res2 = FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=ndarrays_to_parameters(w_client2),
        num_examples=100,
        metrics={"loss": 1.2},
    )

    results = [(MockClientProxy("client_1"), res1), (MockClientProxy("client_2"), res2)]

    aggregated_params, metrics = strategy.aggregate_fit(server_round=1, results=results, failures=[])

    assert aggregated_params is not None
    # Verify trust score for client_2 was clipped to 0.0 by ReLU
    trust_scores = strategy.round_trust_scores[1]
    assert trust_scores["client_1"] > 0.9
    assert trust_scores["client_2"] == 0.0  # Clipped by ReLU

    # Aggregated weights should be positive because client_1 dominated trust
    agg_w = parameters_to_ndarrays(aggregated_params)[0]
    assert (agg_w > 0.0).all()


def test_server_simulation_end_to_end():
    """Verify that run_simulation executes 2 rounds and returns valid history."""
    history = run_simulation(num_rounds=2, num_clients=2, beta=0.7)
    assert history is not None
    # Verify distributed losses were recorded for 2 rounds
    assert len(history.losses_distributed) == 2
