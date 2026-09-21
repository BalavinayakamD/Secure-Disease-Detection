"""
Centralized baseline: train CardioMLP on the complete training set.

This is the upper bound every federated experiment is measured against --
one model, all the data, no privacy noise, no aggregation.

Usage:
    python experiments/01_central_baseline.py [--epochs 30]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocessing import load_processed_data
from src.models.cardio_mlp import build_from_config
from src.utils.metrics import binary_metrics

RESULTS_DIR = 'outputs/results'
MODELS_DIR = 'outputs/models'


def to_tensors(X, y):
    # torch.tensor (not from_numpy): pandas 3.0 arrays are read-only.
    return (
        torch.tensor(X.to_numpy(dtype=np.float32)),
        torch.tensor(y.to_numpy(dtype=np.float32)).unsqueeze(1),
    )


@torch.no_grad()
def evaluate(model, X, y):
    model.eval()
    probs = torch.sigmoid(model(X)).squeeze(1).numpy()
    return binary_metrics(y.squeeze(1).numpy().astype(int), probs)


def main():
    with open('config.yaml') as f:
        cfg = yaml.safe_load(f)

    t = cfg['training']
    parser = argparse.ArgumentParser()
    # config's local_epochs=2 is a FEDERATED knob (epochs per round per client).
    # A centralized run needs real epochs, so it gets its own flag.
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=t['batch_size'])
    parser.add_argument('--lr', type=float, default=t['learning_rate'])
    args = parser.parse_args()

    seed = cfg['data']['random_seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, X_test, y_train, y_test = load_processed_data(cfg['data']['processed_dir'])
    Xtr, ytr = to_tensors(X_train, y_train)
    Xte, yte = to_tensors(X_test, y_test)
    print(f'Train: {tuple(Xtr.shape)}   Test: {tuple(Xte.shape)}')
    print(f'Epochs: {args.epochs} (centralized; config.local_epochs={t["local_epochs"]} '
          f'is for federated rounds)\n')

    loader = DataLoader(
        TensorDataset(Xtr, ytr), batch_size=args.batch_size, shuffle=True
    )

    model = build_from_config(cfg)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    best = {'roc_auc': -1.0}
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)

        train_loss = total / len(Xtr)
        metrics = evaluate(model, Xte, yte)
        history.append({'epoch': epoch, 'train_loss': train_loss, **metrics})

        if metrics['roc_auc'] > best['roc_auc']:
            best = {'epoch': epoch, **metrics}
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(f'epoch {epoch:3d}  loss {train_loss:.4f}  '
              f'acc {metrics["accuracy"]:.4f}  f1 {metrics["f1"]:.4f}  '
              f'auc {metrics["roc_auc"]:.4f}')

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, 'central_baseline.pth')
    torch.save(best_state, model_path)

    results = {
        'experiment': 'centralized_baseline',
        'best': best,
        'final': history[-1],
        'config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'seed': seed,
            'model': cfg['model'],
            'n_train': len(Xtr),
            'n_test': len(Xte),
        },
        'history': history,
    }
    results_path = os.path.join(RESULTS_DIR, 'central_baseline.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\nBest epoch {best["epoch"]}: '
          f'acc {best["accuracy"]:.4f}  f1 {best["f1"]:.4f}  auc {best["roc_auc"]:.4f}')
    print(f'Saved {model_path}')
    print(f'Saved {results_path}')

    if best['accuracy'] < 0.65:
        print('\nWARNING: accuracy below 0.65 -- expected ~0.72-0.74 on this dataset. '
              'Check the scaler and label column before trusting this baseline.')


if __name__ == '__main__':
    main()
