"""
Flower NumPyClient implementation for AT-FedBN.

Implements:
1. FedBN Parameter Localization: Only non-BatchNorm layers (Linear layers: fc1, fc2, fc3)
   are exchanged with the central server. Local BatchNorm parameters (bn1, bn2) and
   running statistics remain on the client for personalization.
2. Lightweight Differential Privacy (DP): Clips L2 norm of weight updates (ΔW) to C
   and adds Gaussian noise N(0, σ²C²) before transmission.
3. Local training and evaluation routines compatible with PyTorch and Flower simulation.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path when executed directly
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

try:
    from src.models.cardio_mlp import CardioMLP
except ModuleNotFoundError:
    from ..models.cardio_mlp import CardioMLP


def load_default_config(config_path: Optional[str] = None) -> dict:
    """Loads configuration from config.yaml if available, else returns standard defaults."""
    if config_path is None:
        potential_path = Path(__file__).resolve().parents[2] / "config.yaml"
        if potential_path.exists():
            config_path = str(potential_path)

    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}

    return {
        "model": {"input_dim": 11, "hidden_dims": [64, 32], "dropout": 0.2, "bn_layers": ["bn1", "bn2"]},
        "training": {"local_epochs": 2, "batch_size": 64, "learning_rate": 0.001, "optimizer": "adam", "loss_fn": "bce"},
        "privacy": {"enabled": True, "clip_norm": 1.0, "noise_scale": 0.01},
    }


class FlowerClient(fl.client.NumPyClient):
    """
    Flower NumPyClient for AT-FedBN architecture.
    """

    def __init__(
        self,
        cid: Union[str, int],
        model: Optional[nn.Module] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        config: Optional[dict] = None,
        device: Optional[torch.device] = None,
        fedbn: bool = True,
    ):
        self.cid = str(cid)
        self.config = config if config is not None else load_default_config()

        # Target compute device
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # FedBN mode vs Standard FedAvg mode
        fed_cfg = self.config.get("federated", {})
        if "fedbn" in fed_cfg:
            self.fedbn = bool(fed_cfg["fedbn"])
        elif fed_cfg.get("strategy") == "fedavg":
            self.fedbn = False
        else:
            self.fedbn = fedbn

        # Initialize model
        if model is not None:
            self.model = model.to(self.device)
        else:
            model_cfg = self.config.get("model", {})
            self.model = CardioMLP(
                input_dim=model_cfg.get("input_dim", 11),
                hidden_dims=model_cfg.get("hidden_dims", [64, 32]),
                dropout=model_cfg.get("dropout", 0.2),
                bn_layers=model_cfg.get("bn_layers", ["bn1", "bn2"]),
            ).to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        # FedBN configuration: if fedbn is False, no layers are kept local
        model_cfg = self.config.get("model", {})
        if self.fedbn:
            self.bn_layer_names = list(model_cfg.get("bn_layers", ["bn1", "bn2"]))
        else:
            self.bn_layer_names = []

        # Privacy configuration: disabled by default in standard FedAvg baseline
        privacy_cfg = self.config.get("privacy", {})
        if not self.fedbn and "enabled" not in privacy_cfg:
            self.dp_enabled = False
        else:
            self.dp_enabled = privacy_cfg.get("enabled", True) if self.fedbn else False

        self.clip_norm = float(privacy_cfg.get("clip_norm", 1.0))
        self.noise_scale = float(privacy_cfg.get("noise_scale", 0.01))

        # Training configuration
        train_cfg = self.config.get("training", {})
        self.local_epochs = int(train_cfg.get("local_epochs", 2))
        self.learning_rate = float(train_cfg.get("learning_rate", 0.001))

        # Cache shared parameter names in deterministic order
        self.shared_param_names = self._get_shared_param_names()

        # Cache initial shared weights received from server to calculate ΔW for DP clipping
        self._initial_shared_params: Optional[List[np.ndarray]] = None

    def _is_bn_param(self, name: str) -> bool:
        """Checks whether a parameter/buffer belongs to a local BatchNorm layer."""
        if not self.fedbn:
            return False
        return any(bn in name for bn in self.bn_layer_names)

    def _get_shared_param_names(self) -> List[str]:
        """Returns non-BatchNorm trainable parameter names to share with the server."""
        return [name for name, _ in self.model.named_parameters() if not self._is_bn_param(name)]

    def get_parameters(self, config: Dict[str, fl.common.Scalar]) -> List[np.ndarray]:
        """
        Extracts shared parameters (non-BN) and applies Lightweight DP if enabled.
        FedBN Rule: BN layers (bn1, bn2) remain strictly local and are omitted.
        Lightweight DP: Clips L2 norm of update (ΔW) to C and adds Gaussian noise N(0, σ²C²).
        """
        state_dict = self.model.state_dict()
        current_shared_tensors = [state_dict[name].detach().clone().cpu() for name in self.shared_param_names]

        # If DP is disabled or no initial parameters are cached, return unperturbed weights
        if not self.dp_enabled or self._initial_shared_params is None:
            return [t.numpy() for t in current_shared_tensors]

        # Calculate weight updates: ΔW = W_current - W_initial
        deltas = []
        for curr, init_arr in zip(current_shared_tensors, self._initial_shared_params):
            init_tensor = torch.from_numpy(init_arr)
            deltas.append(curr - init_tensor)

        # Compute global L2 norm across all shared layers: ||ΔW||_2
        total_norm_sq = sum(torch.sum(d ** 2).item() for d in deltas)
        total_norm = np.sqrt(total_norm_sq)

        # Clip update: ΔW_clipped = ΔW * min(1, C / (||ΔW||_2 + eps))
        clip_coef = min(1.0, self.clip_norm / (total_norm + 1e-8))
        clipped_deltas = [d * clip_coef for d in deltas]

        # Add Gaussian noise: N(0, σ²C²)
        noise_std = self.noise_scale * self.clip_norm
        private_parameters = []
        for d, init_arr in zip(clipped_deltas, self._initial_shared_params):
            noise = torch.randn_like(d) * noise_std
            perturbed_delta = d + noise
            # W_private = W_initial + ΔW_clipped_noised
            reconstructed_weight = torch.from_numpy(init_arr) + perturbed_delta
            private_parameters.append(reconstructed_weight.numpy())

        return private_parameters

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Updates the model's non-BN parameters using weights aggregated by the server.
        Local BatchNorm parameters (affine weights, biases, and running stats) are preserved.
        """
        if len(parameters) != len(self.shared_param_names):
            raise ValueError(
                f"Expected {len(self.shared_param_names)} parameters, but received {len(parameters)}."
            )

        # Cache incoming parameters as initial weights for ΔW calculation
        self._initial_shared_params = [np.copy(p) for p in parameters]

        # Update model state dict for non-BN layers only
        state_dict = self.model.state_dict()
        for name, param in zip(self.shared_param_names, parameters):
            tensor_param = torch.from_numpy(param).to(self.device)
            state_dict[name] = tensor_param

        # Strict=False ensures local BatchNorm statistics are left intact
        self.model.load_state_dict(state_dict, strict=False)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, fl.common.Scalar]
    ) -> Tuple[List[np.ndarray], int, Dict[str, fl.common.Scalar]]:
        """
        Trains the model on the local dataset using FedBN and returns updated parameters.
        """
        self.set_parameters(parameters)

        epochs = int(config.get("local_epochs", self.local_epochs))
        lr = float(config.get("learning_rate", self.learning_rate))

        if self.train_loader is None or len(self.train_loader.dataset) == 0:
            # Fallback for stub/uninitialized client
            return self.get_parameters(config), 0, {"loss": 0.0, "accuracy": 0.0}

        self.model.train()
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        total_loss = 0.0
        correct = 0
        total_samples = 0

        for _ in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for x_batch, y_batch in self.train_loader:
                x_batch = x_batch.to(self.device).float()
                y_batch = y_batch.to(self.device).float().unsqueeze(1) if y_batch.ndim == 1 else y_batch.to(self.device).float()

                optimizer.zero_grad()
                preds = self.model(x_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()

                batch_size = x_batch.size(0)
                epoch_loss += loss.item() * batch_size
                predicted_labels = (preds >= 0.5).float()
                epoch_correct += (predicted_labels == y_batch).sum().item()
                epoch_samples += batch_size

            total_loss += epoch_loss
            correct += epoch_correct
            total_samples += epoch_samples

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = correct / max(total_samples, 1)
        num_examples = len(self.train_loader.dataset)

        metrics = {
            "loss": float(avg_loss),
            "accuracy": float(accuracy),
        }

        return self.get_parameters(config), num_examples, metrics

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, fl.common.Scalar]
    ) -> Tuple[float, int, Dict[str, fl.common.Scalar]]:
        """
        Evaluates the current model on the local validation dataset.
        """
        self.set_parameters(parameters)

        if self.val_loader is None or len(self.val_loader.dataset) == 0:
            return 0.0, 0, {"accuracy": 0.0, "f1": 0.0}

        self.model.eval()
        criterion = nn.BCELoss()

        total_loss = 0.0
        all_preds = []
        all_targets = []
        num_examples = len(self.val_loader.dataset)

        with torch.no_grad():
            for x_batch, y_batch in self.val_loader:
                x_batch = x_batch.to(self.device).float()
                y_batch = y_batch.to(self.device).float().unsqueeze(1) if y_batch.ndim == 1 else y_batch.to(self.device).float()

                preds = self.model(x_batch)
                loss = criterion(preds, y_batch)

                total_loss += loss.item() * x_batch.size(0)
                predicted_labels = (preds >= 0.5).float()

                all_preds.extend(predicted_labels.cpu().squeeze().tolist())
                all_targets.extend(y_batch.cpu().squeeze().tolist())

        avg_loss = float(total_loss / max(num_examples, 1))

        # Metric calculation
        targets_arr = np.array(all_targets, dtype=int)
        preds_arr = np.array(all_preds, dtype=int)

        accuracy = float(np.mean(targets_arr == preds_arr)) if len(targets_arr) > 0 else 0.0

        # F1 Score calculation
        tp = np.sum((targets_arr == 1) & (preds_arr == 1))
        fp = np.sum((targets_arr == 0) & (preds_arr == 1))
        fn = np.sum((targets_arr == 1) & (preds_arr == 0))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        metrics = {
            "accuracy": accuracy,
            "f1": float(f1),
        }

        return avg_loss, num_examples, metrics


def create_client(
    cid: Union[str, int],
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    config: Optional[dict] = None,
) -> fl.client.Client:
    """
    Factory function to create a Flower Client instance for simulation (`flwr.simulation`).
    """
    client = FlowerClient(
        cid=cid,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )
    return client.to_client()


if __name__ == "__main__":
    from torch.utils.data import TensorDataset

    print("[*] Running sanity self-test for FlowerClient (AT-FedBN)...")
    # Generate synthetic tabular dataset with 11 features matching config.yaml
    x_train = torch.randn(128, 11)
    y_train = torch.randint(0, 2, (128, 1)).float()
    x_val = torch.randn(32, 11)
    y_val = torch.randint(0, 2, (32, 1)).float()

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=32)

    client = FlowerClient(cid="hospital_1", train_loader=train_loader, val_loader=val_loader)
    print(f"[*] Client created with ID: {client.cid}")
    print(f"[*] Shared parameters: {client.shared_param_names}")

    init_params = client.get_parameters(config={})
    print(f"[*] Number of shared parameter arrays: {len(init_params)}")

    # Test Fit
    print("[*] Testing fit()...")
    updated_params, num_samples, metrics = client.fit(init_params, config={"local_epochs": 1})
    print(f"[*] Fit completed. Samples: {num_samples}, Metrics: {metrics}")

    # Test Evaluate
    print("[*] Testing evaluate()...")
    eval_loss, eval_samples, eval_metrics = client.evaluate(updated_params, config={})
    print(f"[*] Evaluate completed. Loss: {eval_loss:.4f}, Samples: {eval_samples}, Metrics: {eval_metrics}")
    print("[+] FlowerClient self-test passed successfully!")
