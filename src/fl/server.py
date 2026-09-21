"""
Flower Server & Ray-based Simulation Runner for AT-FedBN.

Coordinates multi-hospital federated training using Ray simulation.
Executes ATFedBNStrategy, tracks client trust scores, and aggregates non-BN parameters.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Union

# Ensure project root is on sys.path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import flwr as fl
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerConfig
from flwr.simulation import start_simulation
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

from src.fl.client import FlowerClient
from src.fl.strategy import ATFedBNStrategy
from src.models.cardio_mlp import CardioMLP


def load_config(config_path: Optional[str] = None) -> dict:
    """Loads YAML configuration file."""
    if config_path is None:
        potential_path = Path(project_root) / "config.yaml"
        if potential_path.exists():
            config_path = str(potential_path)

    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}

    return {
        "data": {"num_clients": 3},
        "model": {"input_dim": 11, "hidden_dims": [64, 32], "dropout": 0.2, "bn_layers": ["bn1", "bn2"]},
        "federated": {"num_rounds": 10, "beta": 0.7, "trust_floor": 0.0, "trust_epsilon": 1e-8},
        "training": {"local_epochs": 2, "batch_size": 64, "learning_rate": 0.001},
        "privacy": {"enabled": True, "clip_norm": 1.0, "noise_scale": 0.01},
    }


def get_initial_parameters(model: torch.nn.Module, bn_layer_names: list) -> fl.common.Parameters:
    """Extracts non-BatchNorm parameters from the initial model as starting server parameters."""
    shared_weights = [
        val.detach().cpu().numpy()
        for name, val in model.named_parameters()
        if not any(bn in name for bn in bn_layer_names)
    ]
    return ndarrays_to_parameters(shared_weights)


def generate_synthetic_hospital_data(input_dim: int = 11, num_samples: int = 128, batch_size: int = 32):
    """Generates synthetic tabular DataLoader for client simulation if real dataset is not yet processed."""
    x = torch.randn(num_samples, input_dim)
    y = torch.randint(0, 2, (num_samples, 1)).float()
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def create_client_fn(config: dict):
    """
    Returns a client_fn compatible with both Flower Context (1.30+) and legacy string CID.
    """
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    input_dim = model_cfg.get("input_dim", 11)
    batch_size = train_cfg.get("batch_size", 32)

    def client_fn(context_or_cid: Union[Context, str, int]) -> fl.client.Client:
        # Extract client ID across Flower versions
        if isinstance(context_or_cid, str):
            cid = context_or_cid
        elif hasattr(context_or_cid, "node_config") and "partition-id" in context_or_cid.node_config:
            cid = str(context_or_cid.node_config["partition-id"])
        elif hasattr(context_or_cid, "node_id"):
            cid = str(context_or_cid.node_id)
        else:
            cid = str(context_or_cid)

        # In future milestones, check data/processed/hospital_{cid}.csv
        # For now, initialize DataLoader with synthetic tabular data matching input_dim
        train_loader = generate_synthetic_hospital_data(input_dim=input_dim, num_samples=128, batch_size=batch_size)
        val_loader = generate_synthetic_hospital_data(input_dim=input_dim, num_samples=32, batch_size=batch_size)

        client = FlowerClient(
            cid=cid,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
        )
        return client.to_client()

    return client_fn


def run_simulation(
    num_rounds: Optional[int] = None,
    num_clients: Optional[int] = None,
    beta: Optional[float] = None,
    config_path: Optional[str] = None,
) -> fl.server.history.History:
    """
    Runs Flower Ray-based simulation for AT-FedBN.
    """
    config = load_config(config_path)

    # Overrides from arguments if provided
    fed_cfg = config.get("federated", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})

    rounds = num_rounds if num_rounds is not None else fed_cfg.get("num_rounds", 10)
    clients_count = num_clients if num_clients is not None else data_cfg.get("num_clients", 3)
    trust_beta = beta if beta is not None else fed_cfg.get("beta", 0.7)

    print(f"[*] Initializing AT-FedBN Simulation: {rounds} rounds, {clients_count} clients, β={trust_beta}")

    # Build reference model and extract non-BN initial parameters
    reference_model = CardioMLP(
        input_dim=model_cfg.get("input_dim", 11),
        hidden_dims=model_cfg.get("hidden_dims", [64, 32]),
        dropout=model_cfg.get("dropout", 0.2),
        bn_layers=model_cfg.get("bn_layers", ["bn1", "bn2"]),
    )
    initial_parameters = get_initial_parameters(
        reference_model, bn_layer_names=model_cfg.get("bn_layers", ["bn1", "bn2"])
    )

    # Instantiate custom AT-FedBN Strategy
    strategy = ATFedBNStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=clients_count,
        min_evaluate_clients=clients_count,
        min_available_clients=clients_count,
        beta=trust_beta,
        trust_floor=fed_cfg.get("trust_floor", 0.0),
        trust_epsilon=fed_cfg.get("trust_epsilon", 1e-8),
        initial_parameters=initial_parameters,
    )

    client_fn = create_client_fn(config)

    # Run simulation using Flower's Virtual Client Engine (Ray)
    history = start_simulation(
        client_fn=client_fn,
        num_clients=clients_count,
        config=ServerConfig(num_rounds=rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )

    print("[+] Simulation completed successfully!")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AT-FedBN Simulation Server")
    parser.add_argument("--rounds", type=int, default=None, help="Number of federated rounds")
    parser.add_argument("--num-clients", type=int, default=None, help="Number of simulated clients")
    parser.add_argument("--beta", type=float, default=None, help="Hybrid weight factor β")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")

    args = parser.parse_args()
    run_simulation(
        num_rounds=args.rounds,
        num_clients=args.num_clients,
        beta=args.beta,
        config_path=args.config,
    )
