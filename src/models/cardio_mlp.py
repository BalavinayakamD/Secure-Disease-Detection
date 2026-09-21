# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
from typing import List, Optional


class CardioMLP(nn.Module):
    """
    3-Layer Multi-Layer Perceptron (MLP) for Tabular Disease Classification.
    Configured for FedBN (Federated Batch Normalization) where BatchNorm layers
    (bn1, bn2) remain strictly local to client hospitals, and Linear layers
    (fc1, fc2, fc3) are aggregated globally.
    """

    def __init__(
        self,
        input_dim: int = 11,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        bn_layers: Optional[List[str]] = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]
        if bn_layers is None:
            bn_layers = ["bn1", "bn2"]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.bn_layer_names = list(bn_layers)

        # Layer 1: Linear -> BatchNorm1d -> ReLU -> Dropout
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Layer 2: Linear -> BatchNorm1d -> ReLU -> Dropout
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Layer 3: Linear (Output) -> Sigmoid
        self.fc3 = nn.Linear(hidden_dims[1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning probability in [0, 1].
        """
        x = self.drop1(self.relu1(self.bn1(self.fc1(x))))
        x = self.drop2(self.relu2(self.bn2(self.fc2(x))))
        x = self.sigmoid(self.fc3(x))
        return x

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning raw logits (for numerical stability with BCEWithLogitsLoss).
        """
        x = self.drop1(self.relu1(self.bn1(self.fc1(x))))
        x = self.drop2(self.relu2(self.bn2(self.fc2(x))))
        return self.fc3(x)

    def is_bn_param(self, param_name: str) -> bool:
        """
        Determines if a state_dict key belongs to a local BatchNorm layer.
        """
        return any(bn_name in param_name for bn_name in self.bn_layer_names)

    def get_shared_param_names(self) -> List[str]:
        """
        Returns the ordered list of parameter names shared with the FL server (non-BN layers).
        """
        return [name for name, _ in self.named_parameters() if not self.is_bn_param(name)]

    def get_local_bn_param_names(self) -> List[str]:
        """
        Returns all parameter and buffer names that stay strictly local (BN layers).
        """
        return [name for name in self.state_dict().keys() if self.is_bn_param(name)]
