import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, equation, in_features, hidden_size, out_size):
        super(MLP, self).__init__()
        self.W1 = nn.Parameter(torch.zeros(in_features, hidden_size))
        pass

    def forward(self, x):
        pass
