import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        
        self.enc_lin = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(64**3*1, 2048),
#            nn.BatchNorm1d(2048),
            nn.ELU(),
            nn.Linear(2048, 256),
#            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Linear(256, 32),
#            nn.BatchNorm1d(32),
            nn.ELU(),
        )
        
        
    def forward(self, x):
        x = self.enc_lin(x)
        
        return x

    
