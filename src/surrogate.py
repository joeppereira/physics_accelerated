import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniSAUFNOJEPA(nn.Module):
    """Fourier Neural Operator for Spectral Signal Prediction."""
    def __init__(self, in_dim=8, out_dim=3, width=32):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, width)
        self.fourier = nn.Conv1d(width, width, 1) 
        self.fc_out = nn.Linear(width, out_dim)

    def forward(self, x):
        x = F.gelu(self.fc_in(x))
        x = x.unsqueeze(-1)
        x = F.gelu(self.fourier(x)).squeeze(-1)
        return self.fc_out(x)
