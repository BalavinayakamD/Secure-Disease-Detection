"""
CardioMLP: the shared model for every AT-FedBN experiment.

The BatchNorm modules are deliberately named `bn1` / `bn2` to match
config.yaml:model.bn_layers. FedBN keeps exactly those parameters local,
so the names are part of the contract, not cosmetic.
"""

import torch
import torch.nn as nn


class CardioMLP(nn.Module):
    """
    3-layer MLP for binary cardiovascular disease classification.

        Linear -> BatchNorm1d -> ReLU -> Dropout
        Linear -> BatchNorm1d -> ReLU -> Dropout
        Linear -> raw logit

    The output is an unactivated logit, trained with BCEWithLogitsLoss for
    numerical stability. Apply torch.sigmoid() at evaluation time.
    """

    def __init__(self, input_dim=11, hidden_dims=(64, 32), dropout=0.2):
        super().__init__()
        h1, h2 = hidden_dims

        self.fc1 = nn.Linear(input_dim, h1)
        self.bn1 = nn.BatchNorm1d(h1)
        self.fc2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2)
        self.fc3 = nn.Linear(h2, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        return self.fc3(x)

    @staticmethod
    def bn_keys(state_dict, bn_layers=('bn1', 'bn2')):
        """
        State-dict keys that FedBN must keep local (weights, biases, running
        stats and num_batches_tracked of every BN layer).
        """
        return [k for k in state_dict if k.split('.')[0] in bn_layers]

    @staticmethod
    def shared_keys(state_dict, bn_layers=('bn1', 'bn2')):
        """State-dict keys that get aggregated on the server."""
        return [k for k in state_dict if k.split('.')[0] not in bn_layers]


def build_from_config(cfg):
    """Instantiate a CardioMLP from a parsed config.yaml dict."""
    m = cfg['model']
    return CardioMLP(
        input_dim=m['input_dim'],
        hidden_dims=tuple(m['hidden_dims']),
        dropout=m['dropout'],
    )


if __name__ == '__main__':
    model = CardioMLP()
    out = model(torch.randn(8, 11))
    print(model)
    print(f'\nOutput shape: {tuple(out.shape)}')
    sd = model.state_dict()
    print(f'Local (BN) keys:  {CardioMLP.bn_keys(sd)}')
    print(f'Shared keys:      {CardioMLP.shared_keys(sd)}')
