import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from .blocks import *


class MultiScaleDis2(nn.Module):
    # (1, 32) -> (64, 16) -> (128, 8) -> (1, 8)
    #   |-> (1, 16) -...-> (1, 4)
    #         |-> (1, 8) -...-> (1, 2)
    def __init__(self, input_dim, n_scale=3, n_layer=3, norm='None', sn=True):
        super(MultiScaleDis2, self).__init__()
        ch = 32
        self.Diss = nn.ModuleList()
#        for _ in range(n_scale):
        self.Diss.append(self._make_net1(ch, input_dim, n_layer, norm, sn))
        self.Diss.append(self._make_net1(ch, input_dim, n_layer, norm, sn))
        self.Diss.append(self._make_net1(ch, input_dim, n_layer, norm, sn))


    def _make_net1(self, ch, input_dim, n_layer, norm, sn):
        model = []
        
        model += [spectral_norm(nn.Conv3d(input_dim, 2, kernel_size=3, stride=1, dilation = 2, padding=0, bias=True))]
        model += [nn.ELU()]


        model += [spectral_norm(nn.Conv3d(2, 3, kernel_size=3, stride=1, dilation = 2, padding=0, bias=True))]
        model += [nn.ELU()]

        model += [spectral_norm(nn.Conv3d(3, 3, kernel_size=3, stride=1, dilation = 2, padding=0, bias=True))]
        model += [nn.ELU()]

        model += [spectral_norm(nn.Conv3d(3, 4, kernel_size=3, stride=1, dilation = 2, padding=0, bias=True))]
        model += [nn.ELU()]

        model += [spectral_norm(nn.Conv3d(4, 4, kernel_size=3, stride=1, dilation = 2, padding=0, bias=True))]
        model += [nn.ELU()]

        model += [spectral_norm(nn.Conv3d(4, 5, kernel_size=3, stride=1, dilation = 2, padding=0, bias=True))]
        model += [nn.ELU()]

        model += [spectral_norm(nn.Conv3d(5, 5, kernel_size=3, stride=1, dilation = 2, padding=0, bias=True))]
        model += [nn.ELU()]

        model += [spectral_norm(nn.Conv3d(5, 6, kernel_size=3, stride=1, dilation = 2, padding=0, bias=True))]
        model += [nn.ELU()]

        model += [spectral_norm(nn.Conv3d(6, 6, kernel_size=3, stride=1, dilation = 2, padding=0, bias=True))]
        model += [nn.ELU()]

        model += [spectral_norm(nn.Conv3d(6, 7, kernel_size=3, stride=1, dilation = 2, padding=0, bias=True))]
        model += [nn.ELU()]

        model += [spectral_norm(nn.Conv3d(7, 7, kernel_size=3, stride=1, dilation = 2, padding=0, bias=True))]
        model += [nn.ELU()]

        model += [spectral_norm(nn.Conv3d(7, 8, kernel_size=3, stride=1, dilation = 2, padding=0, bias=True))]
        model += [nn.ELU()]

        model += [spectral_norm(nn.Conv3d(8, 8, kernel_size=3, stride=1, dilation = 2, padding=0, bias=True))]
        model += [nn.ELU()]

        model += [spectral_norm(nn.Conv3d(8, 9, kernel_size=3, stride=1, dilation = 2, padding=0, bias=True))]
        model += [nn.ELU()]

        model += [spectral_norm(nn.Conv3d(9, 10, kernel_size=3, stride=1, dilation = 2, padding=0, bias=True))]
        model += [nn.ELU()]
        
        model += [spectral_norm(nn.Conv3d(10, 10, kernel_size=4, stride=4, dilation = 1, padding=0, bias=True))]
        
        model += [nn.Flatten(start_dim=1)]
        
        model += [spectral_norm(nn.Linear(10, 10))]
        model += [nn.ELU()]
        model += [spectral_norm(nn.Linear(10, 1))]
        model += [nn.Sigmoid()]
        
        return nn.Sequential(*model)





    def _make_net2(self, ch, input_dim, n_layer, norm, sn):
        model = []
        
        model += [spectral_norm(nn.Conv3d(input_dim, 4, kernel_size=3, stride=1, dilation = 2, padding=0, bias=True))]
        model += [nn.ELU()]

        model += [spectral_norm(nn.Conv3d(4, 5, kernel_size=3, stride=1, dilation = 2, padding=0, bias=True))]
        model += [nn.ELU()]

        model += [spectral_norm(nn.Conv3d(5, 6, kernel_size=3, stride=1, dilation = 2, padding=0, bias=True))]
        model += [nn.ELU()]

        model += [spectral_norm(nn.Conv3d(6, 7, kernel_size=3, stride=1, dilation = 2, padding=0, bias=True))]
        model += [nn.ELU()]

        model += [spectral_norm(nn.Conv3d(7, 8, kernel_size=3, stride=1, dilation = 2, padding=0, bias=True))]
        model += [nn.ELU()]

        model += [spectral_norm(nn.Conv3d(8, 9, kernel_size=3, stride=1, dilation = 2, padding=0, bias=True))]
        model += [nn.ELU()]

        model += [spectral_norm(nn.Conv3d(9, 10, kernel_size=3, stride=1, dilation = 2, padding=0, bias=True))]
        model += [nn.ELU()]
        
        model += [spectral_norm(nn.Conv3d(10, 10, kernel_size=4, stride=4, dilation = 1, padding=0, bias=True))]
        
        model += [nn.Flatten(start_dim=1)]
        
        model += [spectral_norm(nn.Linear(10, 10))]
        model += [nn.ELU()]
        model += [spectral_norm(nn.Linear(10, 1))]
        model += [nn.Sigmoid()]
        
        return nn.Sequential(*model)


        
    def _make_net3(self, ch, input_dim, n_layer, norm, sn):
        model = []
        
        model += [spectral_norm(nn.Conv3d(input_dim, 4, kernel_size=3, stride=1, dilation = 3, padding=0, bias=True))]
        model += [nn.ELU()]

        model += [spectral_norm(nn.Conv3d(4, 6, kernel_size=3, stride=1, dilation = 3, padding=0, bias=True))]
        model += [nn.ELU()]

        model += [spectral_norm(nn.Conv3d(6, 8, kernel_size=3, stride=1, dilation = 3, padding=0, bias=True))]
        model += [nn.ELU()]

        model += [spectral_norm(nn.Conv3d(8, 10, kernel_size=3, stride=1, dilation = 3, padding=0, bias=True))]
        model += [nn.ELU()]

        model += [spectral_norm(nn.Conv3d(10, 10, kernel_size=3, stride=1, dilation = 3, padding=0, bias=True))]
        model += [nn.ELU()]
        
        model += [spectral_norm(nn.Conv3d(10, 10, kernel_size=2, stride=2, dilation = 1, padding=0, bias=True))]
        
        model += [nn.Flatten(start_dim=1)]
        
        model += [spectral_norm(nn.Linear(10, 10))]
        model += [nn.ELU()]
        model += [spectral_norm(nn.Linear(10, 1))]
#        model += [nn.Sigmoid()]
        
        return nn.Sequential(*model)                
        

    def forward(self, x):
        outs = []
        for Dis in self.Diss:
            outs.append(Dis(x))
        return torch.cat(outs, dim = 1)



class MultiScaleDis1_sub(nn.Module):

    # (1, 32) -> (64, 16) -> (128, 8) -> (1, 8)

    #   |-> (1, 16) -...-> (1, 4)

    #         |-> (1, 8) -...-> (1, 2)

    def __init__(self, input_dim, n_scale=3, n_layer=3, norm='None', sn=True):

        super(MultiScaleDis1_sub, self).__init__()
        ch = 4

        self.Diss = nn.ModuleList()

#        for _ in range(n_scale):

        self.Diss.append(self._make_net3(ch, input_dim, n_layer, norm, sn))
        self.Diss.append(self._make_net3(ch, input_dim, n_layer, norm, sn))
        self.Diss.append(self._make_net3(ch, input_dim, n_layer, norm, sn))


    def unet_convs(self, in_channels, out_channels, padding=1):
        return nn.Sequential(
            spectral_norm(nn.Conv3d(in_channels, out_channels, 3, 1, padding, padding_mode = 'replicate')),
            nn.ELU(),
            spectral_norm(nn.Conv3d(out_channels, out_channels, 3, 1, padding, padding_mode = 'replicate')),
            nn.ELU()
        )



    def _make_net1(self, ch, input_dim, n_layer, norm, sn):
        model = []       
        model += [
            self.unet_convs(input_dim, 1),
            nn.Sequential(nn.Conv3d(1, 1, 2, 2, 0),
                self.unet_convs(1, 2)),
            nn.Sequential(nn.Conv3d(2, 2, 2, 2, 0))
#                self.unet_convs(2, 4)),
#            nn.Sequential(nn.Conv3d(4, 4, 2, 2, 0))
#                self.unet_convs(16, 32))
            ]
        
        model += [nn.Flatten(start_dim=1)]
        

        model += [spectral_norm(nn.Linear(2*8**3, 16))]
        model += [nn.ELU()]
        model += [spectral_norm(nn.Linear(16, 1))]
        model += [nn.Sigmoid()]
            
        return nn.Sequential(*model)

        
        
    def _make_net2(self, ch, input_dim, n_layer, norm, sn):
        model = []       
        model += [
            self.unet_convs(input_dim, 1),
            nn.Sequential(nn.Conv3d(1, 1, 2, 2, 0),
                self.unet_convs(1, 2)),
            nn.Sequential(nn.Conv3d(2, 2, 2, 2, 0),
                self.unet_convs(2, 4)),
            nn.Sequential(nn.Conv3d(4, 4, 2, 2, 0))
#                self.unet_convs(16, 32))
            ]
        
        model += [nn.Flatten(start_dim=1)]
        

        model += [spectral_norm(nn.Linear(4*4**3, 16))]
        model += [nn.ELU()]
        model += [spectral_norm(nn.Linear(16, 1))]
        model += [nn.Sigmoid()]
            
        return nn.Sequential(*model)

        
        
    def _make_net3(self, ch, input_dim, n_layer, norm, sn):
        model = []       
        model += [
            self.unet_convs(1, ch),
            nn.Sequential(nn.Conv3d(ch, ch, 2, 2, 0),
                self.unet_convs(ch, ch*2)),
            nn.Sequential(nn.Conv3d(ch*2, ch*2, 2, 2, 0),
                self.unet_convs(ch*2, ch*4)),
            nn.Sequential(nn.Conv3d(ch*4, ch*4, 2, 2, 0),
                self.unet_convs(ch*4, ch*8)),
            nn.Sequential(nn.Conv3d(ch*8, ch*8, 2, 2, 0),
                self.unet_convs(ch*8, ch*16))
                 ]
        
        model += [nn.Flatten(start_dim=1)]
        

        model += [spectral_norm(nn.Linear(ch*16*4**3, 16))]
        model += [nn.ELU()]
        model += [spectral_norm(nn.Linear(16, 1))]
        model += [nn.Sigmoid()]
            
        return nn.Sequential(*model)       


    def _make_net4(self, ch, input_dim, n_layer, norm, sn):
        model = []       
        model += [
            nn.Flatten(start_dim=1),
            spectral_norm(nn.Linear(64**3*input_dim, 512)),
            nn.ELU(),
            spectral_norm(nn.Linear(512, 64)),
            nn.ELU(),
            spectral_norm(nn.Linear(64, 8)),
            nn.ELU(),
            spectral_norm(nn.Linear(8, 1)),
            nn.Sigmoid()
            ]
        
            
        return nn.Sequential(*model)     




    def forward(self, x):

        outs = []

        for Dis in self.Diss:

            outs.append(Dis(x))

        return outs


class MultiScaleDis1_add(nn.Module):

    # (1, 32) -> (64, 16) -> (128, 8) -> (1, 8)

    #   |-> (1, 16) -...-> (1, 4)

    #         |-> (1, 8) -...-> (1, 2)

    def __init__(self, input_dim, n_scale=3, n_layer=3, norm='None', sn=True):

        super(MultiScaleDis1_add, self).__init__()
        ch = 4


        self.Diss = nn.ModuleList()

#        for _ in range(n_scale):

        self.Diss.append(self._make_net3(ch, input_dim, n_layer, norm, sn))
        self.Diss.append(self._make_net4(ch, input_dim, n_layer, norm, sn))
        self.Diss.append(self._make_net3(ch, input_dim, n_layer, norm, sn))



    def unet_convs(self, in_channels, out_channels, padding=1):
        return nn.Sequential(
            spectral_norm(nn.Conv3d(in_channels, out_channels, 3, 1, padding, padding_mode = 'replicate')),
            nn.ELU(),
            spectral_norm(nn.Conv3d(out_channels, out_channels, 3, 1, padding, padding_mode = 'replicate')),
            nn.ELU()
        )



    def _make_net1(self, ch, input_dim, n_layer, norm, sn):
        model = []       
        model += [
            self.unet_convs(input_dim, 1),
            nn.Sequential(nn.Conv3d(1, 1, 2, 2, 0),
                self.unet_convs(1, 2)),
            nn.Sequential(nn.Conv3d(2, 2, 2, 2, 0))
#                self.unet_convs(2, 4)),
#            nn.Sequential(nn.Conv3d(4, 4, 2, 2, 0))
#                self.unet_convs(16, 32))
            ]
        
        model += [nn.Flatten(start_dim=1)]
        

        model += [spectral_norm(nn.Linear(2*8**3, 16))]
        model += [nn.ELU()]
        model += [spectral_norm(nn.Linear(16, 1))]
        model += [nn.Sigmoid()]
            
        return nn.Sequential(*model)

        
        
    def _make_net2(self, ch, input_dim, n_layer, norm, sn):
        model = []       
        model += [
            self.unet_convs(input_dim, 1),
            nn.Sequential(nn.Conv3d(1, 1, 2, 2, 0),
                self.unet_convs(1, 2)),
            nn.Sequential(nn.Conv3d(2, 2, 2, 2, 0),
                self.unet_convs(2, 4)),
            nn.Sequential(nn.Conv3d(4, 4, 2, 2, 0))
#                self.unet_convs(16, 32))
            ]
        
        model += [nn.Flatten(start_dim=1)]
        

        model += [spectral_norm(nn.Linear(4*4**3, 16))]
        model += [nn.ELU()]
        model += [spectral_norm(nn.Linear(16, 1))]
        model += [nn.Sigmoid()]
            
        return nn.Sequential(*model)

        
        
    def _make_net3(self, ch, input_dim, n_layer, norm, sn):
        model = []       
        model += [
            self.unet_convs(1, ch),
            nn.Sequential(nn.Conv3d(ch, ch, 2, 2, 0),
                self.unet_convs(ch, ch*2)),
            nn.Sequential(nn.Conv3d(ch*2, ch*2, 2, 2, 0),
                self.unet_convs(ch*2, ch*4)),
            nn.Sequential(nn.Conv3d(ch*4, ch*4, 2, 2, 0),
                self.unet_convs(ch*4, ch*8)),
            nn.Sequential(nn.Conv3d(ch*8, ch*8, 2, 2, 0),
                self.unet_convs(ch*8, ch*16))
                 ]
        
        model += [nn.Flatten(start_dim=1)]
        

        model += [spectral_norm(nn.Linear(ch*16*4**3, 16))]
        model += [nn.ELU()]
        model += [spectral_norm(nn.Linear(16, 1))]
        model += [nn.Sigmoid()]
            
        return nn.Sequential(*model)       


    def _make_net4(self, ch, input_dim, n_layer, norm, sn):
        model = []       
        model += [
            nn.Flatten(start_dim=1),
            spectral_norm(nn.Linear(64**3*input_dim, 512)),
            nn.ELU(),
            spectral_norm(nn.Linear(512, 64)),
            nn.ELU(),
            spectral_norm(nn.Linear(64, 8)),
            nn.ELU(),
            spectral_norm(nn.Linear(8, 1)),
            nn.Sigmoid()
            ]
        
            
        return nn.Sequential(*model)   

    def forward(self, x):

        outs = []

        for Dis in self.Diss:

            outs.append(Dis(x))

        return outs

