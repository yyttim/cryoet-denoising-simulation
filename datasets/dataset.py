#!/usr/bin/env python
# coding: utf-8

import torch
import torch.utils.data as data
from torchvision.transforms import Compose, ToTensor
import pickle
import os, random
import numpy as np
import h5py         



class datasets(data.Dataset):
    def __init__(self, args):
        self.dataroot = args.dataroot
        
        # A
        images_A = os.path.join(self.dataroot, 'data_noisy_missing.h5')
        h5f = h5py.File(images_A,'r') 
        A_data = h5f['dataset_1'][:]
#        A_data = A_data[:A_data.shape[0]-(A_data.shape[0]%128)]
#        A_data =  ((A_data - A_data.min())  / (A_data.max() - A_data.min() + 1e-23)) * 2 - 1
        self.A = np.squeeze(A_data, axis=tuple(i for i in range(1, A_data.ndim) if A_data.shape[i] == 1))
        h5f.close() 

        
        # B
        images_B = os.path.join(self.dataroot, 'data_noisy.h5')
        h5f = h5py.File(images_B,'r') 
        B_data = h5f['dataset_1'][:]
#        B_data = B_data[:B_data.shape[0]-(B_data.shape[0]%128)]
#        B_data =  ((B_data - B_data.min())  / (B_data.max() - B_data.min() + 1e-23)) * 2 - 1
        self.B = np.squeeze(B_data, axis=tuple(i for i in range(1, B_data.ndim) if B_data.shape[i] == 1))
        h5f.close() 
        
        
        self.A_size = len(self.A)
        self.B_size = len(self.B)
        self.dataset_size = max(self.A_size, self.B_size)

        tensor_trans = [ToTensor()]
        self.tensor_trans = Compose(tensor_trans)
        print('A: %d, B: %d images'%(self.A_size, self.B_size))

        

    def load_npy(self, img_name):
        
        img = img_name
        if img.shape != (64, 64, 64):
                img = img.reshape(64, 64, 64)

        return img

    def __getitem__(self, index):
        if self.dataset_size == self.A_size:
            data_A = self.load_npy(self.A[index])
            data_B = self.load_npy(self.B[random.randint(0, self.B_size - 1)])
        else:
            #print('index',index)
            data_A = self.load_npy(self.A[random.randint(0, self.A_size - 1)])
            data_B = self.load_npy(self.B[index])
            
        '''
        op_raw = {'model':{'missing_wedge_angle':0.0, 'SNR':100}, 
        'ctf':{'pix_size':0.72, 'Dz':-0.5, 'voltage':300, 'Cs':2.0, 'sigma':None}}
        data_A = add_noise.noise(1-data_A, op_raw)
        data_A = (data_A - data_A.min()) / (data_A.max() - data_A.min())
        '''

        data_A, data_B = torch.from_numpy(data_A), torch.from_numpy(data_B)
        
        return (data_A.float(),data_B.float())

    def __len__(self):
        return self.dataset_size
