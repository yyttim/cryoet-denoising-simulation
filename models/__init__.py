#!/usr/bin/env python
# coding: utf-8

from .model import VAE

def get_model(args):
    return VAE(args)