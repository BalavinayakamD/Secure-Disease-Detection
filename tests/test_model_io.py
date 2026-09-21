import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.fl.client import FlowerClient
from src.models.cardio_mlp import CardioMLP


def test_cardio_mlp_forward_and_shapes():
    """Verify CardioMLP output shape and probability range."""
    model = CardioMLP(input_dim=11, hidden_dims=[64, 32], dropout=0.2, bn_layers=["bn1", "bn2"])
    batch_size = 16
    x = torch.randn(batch_size, 11)

    # Forward returns probability in [0, 1]
    out = model(x)
    assert out.shape == (batch_size, 1)
    assert (out >= 0.0).all() and (out <= 1.0).all()

    # Forward logits returns unconstrained values
    logits = model.forward_logits(x)
    assert logits.shape == (batch_size, 1)


def test_cardio_mlp_fedbn_param_isolation():
    """Verify that CardioMLP properly separates shared layers from local BN layers."""
    model = CardioMLP(input_dim=11, hidden_dims=[64, 32], bn_layers=["bn1", "bn2"])
    shared = model.get_shared_param_names()
    local_bn = model.get_local_bn_param_names()

    # Shared layers must only contain Linear layers
    expected_shared = [
        "fc1.weight", "fc1.bias",
        "fc2.weight", "fc2.bias",
        "fc3.weight", "fc3.bias"
    ]
    assert shared == expected_shared

    # Local BN must contain bn1 and bn2 parameters and buffers
    assert "bn1.weight" in local_bn
    assert "bn1.bias" in local_bn
    assert "bn1.running_mean" in local_bn
    assert "bn1.running_var" in local_bn
    assert "bn2.weight" in local_bn
    assert "bn2.bias" in local_bn


def test_flower_client_fedbn_set_parameters_preserves_bn():
    """Verify that set_parameters only modifies non-BN layers and preserves local BN state."""
    model = CardioMLP(input_dim=11, hidden_dims=[64, 32], bn_layers=["bn1", "bn2"])

    # Set custom local BatchNorm state
    with torch.no_grad():
        model.bn1.running_mean.fill_(99.0)
        model.bn1.running_var.fill_(5.0)

    client = FlowerClient(cid="hospital_1", model=model)

    # Prepare random parameters for shared layers
    shared_shapes = [p.shape for name, p in model.named_parameters() if not client._is_bn_param(name)]
    dummy_server_params = [np.random.randn(*shape).astype(np.float32) for shape in shared_shapes]

    client.set_parameters(dummy_server_params)

    # Check that BN running stats were untouched
    assert torch.allclose(client.model.bn1.running_mean, torch.tensor(99.0))
    assert torch.allclose(client.model.bn1.running_var, torch.tensor(5.0))

    # Check that fc1.weight was updated to dummy_server_params[0]
    fc1_weight_np = client.model.fc1.weight.detach().cpu().numpy()
    assert np.allclose(fc1_weight_np, dummy_server_params[0])


def test_flower_client_fit_and_evaluate_lifecycle():
    """Verify the full fit and evaluate lifecycle of FlowerClient."""
    x_train = torch.randn(64, 11)
    y_train = torch.randint(0, 2, (64, 1)).float()
    x_val = torch.randn(32, 11)
    y_val = torch.randint(0, 2, (32, 1)).float()

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=16, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=16)

    client = FlowerClient(
        cid="hospital_test",
        train_loader=train_loader,
        val_loader=val_loader,
        config={"training": {"local_epochs": 1, "learning_rate": 0.01}, "privacy": {"enabled": True, "clip_norm": 1.0, "noise_scale": 0.01}},
    )

    initial_params = client.get_parameters(config={})
    assert len(initial_params) == 6  # fc1, fc2, fc3 weights and biases

    # Test fit
    updated_params, num_samples, metrics = client.fit(initial_params, config={"local_epochs": 1})
    assert len(updated_params) == 6
    assert num_samples == 64
    assert "loss" in metrics and "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0

    # Test evaluate
    eval_loss, eval_samples, eval_metrics = client.evaluate(updated_params, config={})
    assert eval_samples == 32
    assert eval_loss >= 0.0
    assert "accuracy" in eval_metrics and "f1" in eval_metrics
    assert 0.0 <= eval_metrics["accuracy"] <= 1.0
    assert 0.0 <= eval_metrics["f1"] <= 1.0


def test_flower_client_lightweight_dp():
    """Verify that Lightweight DP modifies parameters with noise and clipping."""
    torch.manual_seed(42)
    np.random.seed(42)

    x_train = torch.randn(32, 11)
    y_train = torch.randint(0, 2, (32, 1)).float()
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=16)

    # Client with DP enabled
    client_dp = FlowerClient(
        cid="dp_client",
        train_loader=train_loader,
        config={"training": {"local_epochs": 1}, "privacy": {"enabled": True, "clip_norm": 0.5, "noise_scale": 0.05}},
    )
    init_params = client_dp.get_parameters({})
    dp_params, _, _ = client_dp.fit(init_params, {})

    # Client without DP
    client_no_dp = FlowerClient(
        cid="no_dp_client",
        train_loader=train_loader,
        config={"training": {"local_epochs": 1}, "privacy": {"enabled": False}},
    )
    init_params_no_dp = [np.copy(p) for p in init_params]
    # Set same initial parameters
    client_no_dp.set_parameters(init_params_no_dp)
    # Fit
    no_dp_params, _, _ = client_no_dp.fit(init_params_no_dp, {})

    # Because DP adds noise and clipping, returned parameters should differ
    diff = sum(np.sum(np.abs(p1 - p2)) for p1, p2 in zip(dp_params, no_dp_params))
    assert diff > 0.0
