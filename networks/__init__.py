#!/usr/bin/env python
# coding: utf-8

def get_networks(args):
    
    from .discriminator import MultiScaleDis1_sub, MultiScaleDis1_add, MultiScaleDis2
    disc_add = MultiScaleDis1_add(1, args.dis_scale)
    disc_sub = MultiScaleDis1_sub(1, args.dis_scale)
    
    disc_add_spectral = MultiScaleDis2(2, args.dis_scale)
    disc_sub_spectral = MultiScaleDis2(2, args.dis_scale)


    from .denoiser import Dns_joint, Dns_seperate
    dns = Dns_joint()
    #dns = Dns_seperate()

    from .encoder import Encoder
    E = Encoder()

    from .generators import Generator_1
    G1 = Generator_1()
    
    import torch.nn as nn
    
    return nn.DataParallel(disc_add), nn.DataParallel(disc_sub), nn.DataParallel(disc_add_spectral), nn.DataParallel(disc_sub_spectral), nn.DataParallel(E), nn.DataParallel(G1)
