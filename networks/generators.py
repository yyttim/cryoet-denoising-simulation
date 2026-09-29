import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from .blocks import *

from torch.autograd import Variable

class Generator_1(nn.Module):
    def __init__(self):
        super(Generator_1, self).__init__()
        
        self.decoder_lin = nn.Sequential(
            nn.Linear(32, 256),
#            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Linear(256, 2048),
#            nn.BatchNorm1d(2048),
            nn.ELU(),
            nn.Linear(2048, 64**3*1),
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(1, 64, 64, 64))
        
        dec3 = [nn.Conv3d(1, 1, kernel_size=1, stride=1, padding=0)]

#        dec3 = [nn.Conv3d(1, 32, 3, 1, 1, padding_mode = 'replicate'),
#                nn.BatchNorm3d(32),
#                nn.ELU(),
#                nn.Conv3d(32, 32, 3, 1, 1, padding_mode = 'replicate'),
#                nn.BatchNorm3d(32),
#                nn.ELU(),
#                nn.Conv3d(32, 1, 3, 1, 1, padding_mode = 'replicate')
#        ]
        
        self.dec3 = nn.Sequential(*dec3)
        

    def forward(self, x, permute):
        x = self.decoder_lin(x)
        if permute:
            x = x[:, torch.randperm(x.nelement()//x.shape[0])].view(x.size())
        x = self.unflatten(x)
        x = self.dec3(x)

        return x



class Generator_3(nn.Module):
    def __init__(self):
        super(Generator_3, self).__init__()
        
        dec3 = [nn.Conv3d(1, 1, kernel_size=1, stride=1, padding=0)]
        self.dec3 = nn.Sequential(*dec3)
        

    def forward(self, x):
        x = self.dec3(x)
        return x

 

