import torch
from torch import nn

from typing import List


class BinaryClassifier(nn.Module):

    def __init__(self,
                 codes_dim: int,
                 hidden_dims: List[int],
                 k: int):
        assert len(hidden_dims) > 0

        super(BinaryClassifier, self).__init__()

        self.K: int = k
        self.codes_dim: int = codes_dim

        self.class_fc1 = nn.Linear(self.codes_dim, hidden_dims[0])
        
        hidden_modules = []
        for i in range(len(hidden_dims) - 1):
            hidden_modules.append(
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            hidden_modules.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            hidden_modules.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*hidden_modules)
        
        self.class_fc2 = nn.Linear(hidden_dims[-1], 2)

