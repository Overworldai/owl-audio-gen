import torch 
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, dim, dim_middle = None, dim_out = None):
        super().__init__()

        if dim_out is None:
            dim_out = dim
        if dim_middle is None:
            dim_middle = 4 * dim_out
        
        self.fc1 = nn.Linear(dim, dim_middle)
        self.fc2 = nn.Linear(dim_middle, dim_out)

    def forward(self, x):
        x = self.fc1(x)
        x = F.silu(x)
        x = self.fc2(x)
    
        return x