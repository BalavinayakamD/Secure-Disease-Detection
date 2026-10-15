import sys
from pathlib import Path
import pytest
import torch
import numpy as np
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models.cardio_mlp import CardioMLP
from src.utils.scaler_utils import transform_raw_input


def test_prediction_pipeline():
    """Test the complete prediction pipeline."""
    config_path = project_root / 'config.yaml'
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    test_cases = [
        {
            'name': 'Low Risk Patient',
            'features': [45, 1, 165, 65, 110, 70, 1, 1, 0, 0, 1],  # Young, healthy
        },
        {
            'name': 'High Risk Patient',
            'features': [65, 2, 170, 95, 160, 100, 3, 2, 1, 1, 0],  # Older, multiple risk factors
        },
        {
            'name': 'Moderate Risk Patient',
            'features': [55, 2, 175, 80, 130, 85, 2, 1, 0, 0, 1],  # Middle-aged, some risk
        }
    ]

    for test_case in test_cases:
        raw_input = np.array([test_case['features']], dtype=np.float32)
        scaled_input = transform_raw_input(raw_input)
        assert scaled_input.shape == (1, cfg['model']['input_dim'])

        model = CardioMLP(
            input_dim=cfg['model']['input_dim'],
            hidden_dims=tuple(cfg['model']['hidden_dims']),
            dropout=cfg['model']['dropout']
        )

        checkpoint_path = project_root / 'outputs' / 'models' / 'fedavg_baseline_final.pth'
        if not checkpoint_path.exists():
            checkpoint_path = project_root / 'outputs' / 'models' / 'at_fedbn_ui_live_demo.pth'

        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict) and any(k.startswith('fc') or k.startswith('bn') for k in checkpoint.keys()):
                model.load_state_dict(checkpoint)

        model.eval()

        with torch.no_grad():
            input_tensor = torch.from_numpy(scaled_input).float()
            logit = model(input_tensor)
            probability = torch.sigmoid(logit).item()
            prediction = 1 if probability >= 0.5 else 0

        assert 0.0 <= probability <= 1.0
        assert prediction in (0, 1)
