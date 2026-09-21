"""Checks for CardioMLP's shape and its FedBN state-dict contract."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cardio_mlp import CardioMLP, build_from_config


def test_forward_shape():
    model = CardioMLP(input_dim=11)
    assert model(torch.randn(16, 11)).shape == (16, 1)


def test_output_is_a_logit():
    """Raw logits, not probabilities -- BCEWithLogitsLoss depends on this."""
    model = CardioMLP(input_dim=11)
    with torch.no_grad():
        out = model(torch.randn(256, 11) * 20)
    assert out.min() < 0, 'a sigmoid output could never be negative'


def test_state_dict_round_trip():
    a = CardioMLP(input_dim=11)
    b = CardioMLP(input_dim=11)
    b.load_state_dict(a.state_dict())

    a.eval()
    b.eval()
    x = torch.randn(8, 11)
    with torch.no_grad():
        assert torch.allclose(a(x), b(x))


def test_bn_and_shared_keys_partition_the_state_dict():
    """
    FedBN holds BN parameters back and aggregates everything else. If a rename
    ever breaks this split, personalization silently stops working -- so assert it.
    """
    sd = CardioMLP().state_dict()
    bn = CardioMLP.bn_keys(sd)
    shared = CardioMLP.shared_keys(sd)

    assert set(bn) | set(shared) == set(sd)
    assert not set(bn) & set(shared)

    # Both BN layers must contribute affine params AND running stats.
    for layer in ('bn1', 'bn2'):
        for suffix in ('weight', 'bias', 'running_mean', 'running_var'):
            assert f'{layer}.{suffix}' in bn

    assert all(k.startswith(('fc1', 'fc2', 'fc3')) for k in shared)


def test_build_from_config_matches_config_yaml():
    import yaml

    with open('config.yaml') as f:
        cfg = yaml.safe_load(f)

    model = build_from_config(cfg)
    assert model.fc1.in_features == cfg['model']['input_dim']
    assert [model.fc1.out_features, model.fc2.out_features] == cfg['model']['hidden_dims']
    # The BN module names must match what config.yaml tells FedBN to keep local.
    assert sorted(cfg['model']['bn_layers']) == ['bn1', 'bn2']
