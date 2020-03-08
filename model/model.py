"""
Model for estimating data in the UCI Energy Efficiency Dataset 2012.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple 3 layer model for predicting building energy efficiency.
    """

    def __init__(self):
        super().__init__()
        self.norm = nn.BatchNorm1d(8)
        self.fc_1 = nn.Linear(8, 64)
        self.fc_2 = nn.Linear(64, 32)
        self.fc_3 = nn.Linear(32, 2)

    # pylint: disable=arguments-differ
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.fc_1(x)
        x = torch.tanh(x)
        x = self.fc_2(x)
        x = torch.tanh(x)
        x = self.fc_3(x)
        return x
