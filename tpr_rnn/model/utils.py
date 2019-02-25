import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, equation, in_features, hidden_size, out_size):
        super(MLP, self).__init__()
        self.equation = equation
        # Layers
        # 1
        self.W1 = nn.Parameter(torch.zeros(in_features, hidden_size))
        nn.init.xavier_uniform_(self.W1.data)
        self.b1 = nn.parameter(torch.zeros(hidden_size))
        # 2
        self.W2 = nn.Parameter(torch.zeros(hidden_size, out_size))
        nn.init.xavier_uniform_(self.W2.data)
        self.b2 = nn.Parameter(torch.zeros(out_size))

    def forward(self, x):
        hidden = F.tanh(torch.einsum(self.equation, x, self.W1) + self.b1)
        out = F.tanh(torch.einsum(self.equation, hidden, self.W2) + self.b2)
        return out
