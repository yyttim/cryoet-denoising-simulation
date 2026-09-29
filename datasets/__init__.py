#!/usr/bin/env python
# coding: utf-8

import torch
from torch.utils.data import DataLoader

def get_dataloader(args):
    
    from .dataset import datasets

    dataset = datasets(args)
    loader = DataLoader(dataset, 
                        batch_size=args.batch_size, 
                        shuffle=True, 
                        num_workers=args.num_workers)

    return loader
