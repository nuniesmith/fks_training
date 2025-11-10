"""
Shared Backbone Network for PPO

Feature extraction network shared between actor and critic heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class BackboneNetwork(nn.Module):
    """Shared backbone network for feature extraction"""
    
    def __init__(
        self,
        in_features: int = 22,
        hidden_dimensions: int = 128,
        out_features: int = 64,
        dropout: float = 0.2
    ):
        super().__init__()
        self.layer1 = nn.Linear(in_features, hidden_dimensions)
        self.layer2 = nn.Linear(hidden_dimensions, hidden_dimensions)
        self.layer3 = nn.Linear(hidden_dimensions, out_features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through shared backbone"""
        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        x = F.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x

