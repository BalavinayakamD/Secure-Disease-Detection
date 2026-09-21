"""
Custom Flower Strategy: ATFedBNStrategy (Adaptive Trust-Weighted Federated Batch Normalization).

Implements:
1. Update Consistency Trust Mechanism:
   - Evaluates client updates (ΔW_i) against global update direction (ΔW_global) using Cosine Similarity.
   - Applies ReLU clipping to zero-out negative or diverging updates.
2. Hybrid Adaptive Aggregation:
   - FinalWeight_i = (β * VolumeWeight_i) + ((1 - β) * TrustWeight_i)
   - Aggregates non-BatchNorm parameters according to the hybrid weights.
"""

import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

# Ensure project root is on sys.path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import flwr as fl
from flwr.common import (
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
import numpy as np


class ATFedBNStrategy(FedAvg):
    """
    Adaptive Trust-Weighted Federated Batch Normalization Strategy (AT-FedBN).
    """

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        beta: float = 0.7,
        trust_floor: float = 0.0,
        trust_epsilon: float = 1e-8,
        initial_parameters: Optional[Parameters] = None,
        evaluate_fn: Optional[
            Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        fit_metrics_aggregation_fn: Optional[Callable[[List[Tuple[int, Dict[str, Scalar]]]], Dict[str, Scalar]]] = None,
        evaluate_metrics_aggregation_fn: Optional[Callable[[List[Tuple[int, Dict[str, Scalar]]]], Dict[str, Scalar]]] = None,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            initial_parameters=initial_parameters,
        )
        self.beta = beta
        self.trust_floor = trust_floor
        self.trust_epsilon = trust_epsilon

        # Track the latest global non-BN model parameters
        self.current_weights: Optional[NDArrays] = (
            parameters_to_ndarrays(initial_parameters) if initial_parameters is not None else None
        )

        # Track historical trust scores per round
        self.round_trust_scores: Dict[int, Dict[str, float]] = {}

    def compute_cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Computes cosine similarity between two 1D vectors."""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < self.trust_epsilon or norm2 < self.trust_epsilon:
            return 1.0
        dot_product = np.dot(v1, v2)
        sim = dot_product / (norm1 * norm2 + self.trust_epsilon)
        return float(np.clip(sim, -1.0, 1.0))

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregates client model updates using Update Consistency Trust Scores
        and Hybrid Volume-Trust Normalization.
        """
        if not results:
            return None, {}

        if not self.accept_failures and failures:
            return None, {}

        # 1. Unpack client weights, sample counts, and identifiers
        client_weights_list: List[NDArrays] = []
        client_samples: List[int] = []
        client_cids: List[str] = []

        for client_proxy, fit_res in results:
            client_weights = parameters_to_ndarrays(fit_res.parameters)
            client_weights_list.append(client_weights)
            client_samples.append(fit_res.num_examples)
            client_cids.append(client_proxy.cid)

        total_samples = sum(client_samples)
        num_clients = len(results)

        # If this is round 1 and no initial parameters were provided, initialize from mean weights
        if self.current_weights is None:
            self.current_weights = [
                np.mean([weights[i] for weights in client_weights_list], axis=0)
                for i in range(len(client_weights_list[0]))
            ]

        # 2. Compute client parameter deltas: ΔW_i = W_i - W_global(t-1)
        client_flat_deltas: List[np.ndarray] = []
        for client_weights in client_weights_list:
            delta_layers = [
                w_curr - w_prev
                for w_curr, w_prev in zip(client_weights, self.current_weights)
            ]
            flat_delta = np.concatenate([layer.ravel() for layer in delta_layers])
            client_flat_deltas.append(flat_delta)

        # 3. Compute volume weights and global reference delta: ΔW_global
        volume_weights = np.array(
            [n / max(total_samples, 1) for n in client_samples], dtype=np.float64
        )
        global_flat_delta = sum(
            vw * delta for vw, delta in zip(volume_weights, client_flat_deltas)
        )

        # 4. Compute Trust Score via ReLU(CosineSimilarity(ΔW_i, ΔW_global))
        raw_trust_scores: List[float] = []
        for flat_delta in client_flat_deltas:
            cos_sim = self.compute_cosine_similarity(flat_delta, global_flat_delta)
            # ReLU clipping: clips negative similarities to trust_floor (0.0)
            trust = max(self.trust_floor, cos_sim)
            raw_trust_scores.append(trust)

        sum_trust = sum(raw_trust_scores)
        if sum_trust > self.trust_epsilon:
            trust_weights = np.array(
                [t / sum_trust for t in raw_trust_scores], dtype=np.float64
            )
        else:
            # Fallback to volume weights if all trust scores are zeroed
            trust_weights = np.copy(volume_weights)

        # 5. Hybrid Weight Normalization: (β * VolumeWeight) + ((1 - β) * TrustWeight)
        final_weights = (self.beta * volume_weights) + ((1.0 - self.beta) * trust_weights)
        final_weights = final_weights / np.sum(final_weights)

        # 6. Aggregate global parameters using final hybrid weights
        num_layers = len(self.current_weights)
        aggregated_weights: NDArrays = []

        for layer_idx in range(num_layers):
            layer_update = np.zeros_like(self.current_weights[layer_idx], dtype=np.float64)
            for client_idx, client_weights in enumerate(client_weights_list):
                layer_update += final_weights[client_idx] * client_weights[layer_idx]
            aggregated_weights.append(layer_update.astype(self.current_weights[layer_idx].dtype))

        # Update cached global weights for next round
        self.current_weights = aggregated_weights

        # Store trust scores for analytics
        round_trust_dict = {
            cid: float(score) for cid, score in zip(client_cids, raw_trust_scores)
        }
        self.round_trust_scores[server_round] = round_trust_dict

        # Collect fit metrics from clients
        metrics_aggregated: Dict[str, Scalar] = {
            "mean_trust_score": float(np.mean(raw_trust_scores)),
            "min_trust_score": float(np.min(raw_trust_scores)),
            "max_trust_score": float(np.max(raw_trust_scores)),
        }
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated.update(self.fit_metrics_aggregation_fn(fit_metrics))

        return ndarrays_to_parameters(aggregated_weights), metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregates distributed evaluation metrics across clients.
        """
        if not results:
            return None, {}

        if not self.accept_failures and failures:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        metrics_aggregated: Dict[str, Scalar] = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        else:
            # Default aggregation for accuracy and f1
            accuracies = [res.metrics.get("accuracy", 0.0) for _, res in results if "accuracy" in res.metrics]
            if accuracies:
                metrics_aggregated["accuracy"] = float(np.mean(accuracies))
            f1s = [res.metrics.get("f1", 0.0) for _, res in results if "f1" in res.metrics]
            if f1s:
                metrics_aggregated["f1"] = float(np.mean(f1s))

        return loss_aggregated, metrics_aggregated


def weighted_loss_avg(results: List[Tuple[int, float]]) -> float:
    """Aggregates loss weighted by sample counts."""
    total_examples = sum(num_examples for num_examples, _ in results)
    weighted_losses = [num_examples * loss for num_examples, loss in results]
    return float(sum(weighted_losses) / max(total_examples, 1))
