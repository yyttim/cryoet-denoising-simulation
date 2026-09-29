#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast

from networks import get_networks
from .NEM import NEM

import numpy as np
import pandas as pd
from PIL import Image

from networks.unet import UNet_generator, UNet_discriminator
import gc



def cov(tensor, rowvar=True, bias=False):
    """Estimate a covariance matrix (np.cov)
    https://gist.github.com/ModarTensai/5ab449acba9df1a26c12060240773110
    """
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    return factor * tensor @ tensor.transpose(-1, -2).conj()



def grid_displacement_to_center(size, mid_co=None):
    """grid functions"""
    size = np.array(size, dtype=float)
    assert size.ndim == 1

    if mid_co is None:
        # IMPORTANT: following python convension, in index starts from 0 to size-1!!!
        # So (siz-1)/2 is real symmetry center of the volume
        mid_co = (np.array(size) - 1) / 2

    if size.size == 3:
        # construct a gauss function whose center is at center of volume
        grid = np.mgrid[0:size[0], 0:size[1], 0:size[2]]

        for dim in range(3):
            grid[dim, :, :, :] -= mid_co[dim]

    elif size.size == 2:
        # construct a gauss function whose center is at center of volume
        grid = np.mgrid[0:size[0], 0:size[1]]

        for dim in range(2):
            grid[dim, :, :] -= mid_co[dim]

    else:
        assert False

    return grid



def fft_mid_co(siz):
    assert all((np.mod(siz, 1) == 0))
    assert all((np.array(siz) > 0))
    mid_co = np.zeros(len(siz))
    for i in range(len(mid_co)):
        m = siz[i]
        mid_co[i] = np.floor((m / 2))
    return mid_co



def tilt_mask(size, tilt_ang1, tilt_ang2=None, tilt_axis=1, light_axis=2):
    """wedge mask defined using tilt angles light axis is the direction of electrons"""
    assert tilt_axis != light_axis

    if tilt_ang2 is None:
        tilt_ang2 = float(np.abs(tilt_ang1))
        tilt_ang1 = -tilt_ang2

    else:
        assert tilt_ang1 < 0
        assert tilt_ang2 > 0

    tilt_ang1 = (tilt_ang1 / 180.0) * np.pi
    tilt_ang2 = (tilt_ang2 / 180.0) * np.pi

    g = grid_displacement_to_center(size=size, mid_co=fft_mid_co(siz=size))

    plane_axis = set([0, 1, 2])
    plane_axis.difference_update([light_axis, tilt_axis])
    assert len(plane_axis) == 1
    plane_axis = list(plane_axis)[0]

    x_light = g[light_axis]
    x_plane = g[plane_axis]

    m = np.zeros(size, dtype=float)

    m[np.logical_and(x_light <= (np.tan(tilt_ang1) * x_plane), x_light >= (np.tan(tilt_ang2) * x_plane))] = 1.0
    m[np.logical_and(x_light >= (np.tan(tilt_ang1) * x_plane), x_light <= (np.tan(tilt_ang2) * x_plane))] = 1.0


    return m



class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.args = args
        self.maskB = tilt_mask([64,64,64], -60, 60, 1, 2)
#        self.maskB = self.maskB + self.maskB.transpose([1,0,2])
#        self.maskB[self.maskB > 1] = 1
        self.maskA = self.maskB.transpose([2,1,0])                
        
        self.maskB = torch.from_numpy(np.tile(self.maskB.reshape(1,1,64,64,64), (1,1,1,1,1))).cuda().detach()
        self.maskA = torch.from_numpy(np.tile(self.maskA.reshape(1,1,64,64,64), (1,1,1,1,1))).cuda().detach()
        
        
        # networks
        self.disc_add, self.disc_sub, self.disc_add_spectral, self.disc_sub_spectral, self.E, self.G1 = get_networks(args)
        self.disc_add = nn.DataParallel(UNet_discriminator(in_channels=1, out_channels=1, init_features = 8)).cuda()
        self.disc_sub = nn.DataParallel(UNet_discriminator(in_channels=1, out_channels=1, init_features = 8)).cuda()
        self.NEM = NEM(self.args)

        self.dns = nn.DataParallel(UNet_generator(in_channels=1, out_channels=1, init_features = 32)).cuda()
        self.G2 = nn.DataParallel(UNet_generator(in_channels=1, out_channels=1, init_features = 32)).cuda()
        
                
        # optimizers
        self.E_opt = torch.optim.AdamW(self.E.parameters(), lr=args.lr, weight_decay=1e-6)
        self.G1_opt = torch.optim.AdamW(self.G1.parameters(), lr=args.lr, weight_decay=1e-6)
        self.G2_opt = torch.optim.AdamW(self.G2.parameters(), lr=args.lr, weight_decay=1e-6)
        self.dns_opt = torch.optim.AdamW(self.dns.parameters(), lr=args.lr, weight_decay=1e-6)
        self.disc_add_opt = torch.optim.AdamW(self.disc_add.parameters(), lr=args.lr, weight_decay=1e-6)
        self.disc_sub_opt = torch.optim.AdamW(self.disc_sub.parameters(), lr=args.lr, weight_decay=1e-6)
        self.disc_add_spectral_opt = torch.optim.AdamW(self.disc_add_spectral.parameters(), lr=args.lr, weight_decay=1e-6)
        self.disc_sub_spectral_opt = torch.optim.AdamW(self.disc_sub_spectral.parameters(), lr=args.lr, weight_decay=1e-6)

        self.setgpu()



    def _train_G(self):
        
        self.A, self.d_A_1, self.d_A_2 = self.NEM.forward(self.oA, self.B, self.E, self.G1, self.G2) #adding noise on oA
        self.oA_rec = self.dns(torch.fft.ifftn(torch.fft.ifftshift(torch.fft.fftshift(torch.fft.fftn(self.A.detach())) * self.maskB)).real.float()) #denoising on A
        self.oB = self.dns(self.B)     #denoising on B
        self.B_rec, self.enc, self.d_B_1, self.d_B_2 = self.NEM.forward_single_enc(torch.fft.ifftn(torch.fft.ifftshift(torch.fft.fftshift(torch.fft.fftn(self.oB.detach())) * self.maskA)).real.float() , self.B, self.E, self.G1, self.G2)              #adding noise on oB

        # adversarial loss
        adv_fake_A, adv_fake_B = self.disc_add.forward(self.A), self.disc_sub.forward(self.oB) 
        adv_fake_A_spectral, adv_fake_B_spectral = self.disc_add_spectral.forward(self.fft(self.A)), self.disc_sub_spectral.forward(self.fft(self.oB))
        self.adv_G_loss = 1e0 * (self.D_loss_inference(adv_fake_A) + self.D_loss_inference(adv_fake_B) + self.D_loss_inference(adv_fake_A_spectral) + self.D_loss_inference(adv_fake_B_spectral))
        
        # reconstruction loss
#        self.rec_loss = 1 * (self.robustL1loss(1e4 * self.B_rec, 1e4 * self.B) + self.robustL1loss(1e4 * self.oA_rec, 1e4 * self.oA) + 1 - nn.CosineSimilarity(dim = -1)(self.B.view(-1,1,32**3), self.d_B.view(-1,1,32**3)).mean() + 1 - nn.CosineSimilarity(dim = -1)(self.B.view(-1,1,32**3), self.oB.view(-1,1,32**3)).mean() + torch.mean(self.d_B).abs() + torch.mean(self.d_A).abs())
        
        self.rec_loss = (self.robustL1loss_masked(self.B_rec, self.B, 1 - self.maskA) + self.robustL1loss_masked(self.oA_rec, self.oA, 1 - self.maskB))# + self.robustL1loss_masked(self.oB, self.B, self.maskA * self.maskB) + self.robustL1loss_masked(self.A, self.oA, self.maskA * self.maskB)) 
          
        # KL divergence
#        enc_mean = torch.mean(self.enc, dim = 0, keepdim = True)    
#        enc_cov = cov(self.enc.T)
#        enc_cov_inv = torch.linalg.inv(enc_cov)
#        d = torch.tensor(enc_cov.shape[0])

#        self.kld_loss = torch.matmul(enc_mean, enc_mean.T) + nn.MSELoss()(enc_cov, torch.eye(enc_cov.shape[1]).cuda())


#        self.kld_loss = torch.minimum(1e-1 * 0.5 * (torch.log(torch.maximum(torch.det(enc_cov), torch.tensor(1e-12))) - torch.log(d) - d + torch.trace(enc_cov_inv) + torch.matmul(torch.matmul(enc_mean, enc_cov_inv), enc_mean.T)), torch.tensor(100))
#        self.enc_var = torch.var(self.enc, dim = 0)

#        self.kld_loss = 1e-3 * -0.5 * torch.sum(1 + torch.log(torch.maximum(self.enc_var, torch.tensor(1e-12))) - torch.mean(self.enc, dim = 0).pow(2) - self.enc_var)

        # Mode seeking loss
        self.ms_loss = 1 * self.correlationloss(self.d_A_1, self.d_A_2)

        # total loss
        loss = self.adv_G_loss +  1e0 * self.rec_loss + self.ms_loss #+ self.kld_loss 
        
        self.loss = loss
    
#        self.E_opt.zero_grad()
#        self.G1_opt.zero_grad()
        self.G2_opt.zero_grad()
        self.dns_opt.zero_grad()
        loss.backward()
#        self.E_opt.step()
#        self.G1_opt.step()
        self.G2_opt.step()
        self.dns_opt.step()
        
        
    def _train_D(self):

        self.A, _ = self.NEM.forward_single(self.oA, self.B, self.E, self.G1, self.G2) # adding noise on oA
        self.oB = self.dns(self.B)     # denoising on B
        
        # adversarial loss
        adv_real_add, adv_fake_add = self.disc_add.forward(self.B), self.disc_add.forward(self.A)
        adv_real_add_spectral, adv_fake_add_spectral = self.disc_add_spectral.forward(self.fft(self.B)), self.disc_add_spectral.forward(self.fft(self.A))
        self.adv_D_loss_add = self.D_loss_train(adv_real_add, adv_fake_add) + self.D_loss_train(adv_real_add_spectral, adv_fake_add_spectral)

        adv_real_sub, adv_fake_sub = self.disc_sub.forward(self.oA), self.disc_sub.forward(self.oB)
        adv_real_sub_spectral, adv_fake_sub_spectral = self.disc_sub_spectral.forward(self.fft(self.oA)), self.disc_sub_spectral.forward(self.fft(self.oB))
        self.adv_D_loss_sub = self.D_loss_train(adv_real_sub, adv_fake_sub) + self.D_loss_train(adv_real_sub_spectral, adv_fake_sub_spectral)
        
        self.adv_D_loss = 1e0 * (self.adv_D_loss_add + self.adv_D_loss_sub)
        
        
        self.disc_add_opt.zero_grad()
        self.disc_sub_opt.zero_grad()
        self.disc_add_spectral_opt.zero_grad()
        self.disc_sub_spectral_opt.zero_grad()
        self.adv_D_loss.backward()
        self.disc_add_opt.step()
        self.disc_sub_opt.step()
        self.disc_add_spectral_opt.step()
        self.disc_sub_spectral_opt.step()
        



    def forward(self, ori_images_a, images_b):
        self.oA, self.B = ori_images_a.float(), images_b.float()
        
        self._train_D() #training discriminator
        del self.A, self.oB 
        gc.collect()
        torch.cuda.empty_cache()
        self._train_G() #training generator


####################################################################
#---------------------------- Tools --------------------------------
####################################################################

    def setgpu(self):
        self.E.cuda()
        self.G1.cuda()
        self.G2.cuda()
        self.dns.cuda()
        self.disc_add.cuda()
        self.disc_sub.cuda()
        self.disc_add_spectral.cuda()
        self.disc_sub_spectral.cuda()
        
        
    def fft(self, x):
        x_fft = torch.fft.fftshift(torch.fft.fftn(x, dim=(-3, -2, -1)), dim=(-3, -2, -1))
        x_fft = torch.cat([x_fft.real, x_fft.imag], 1).float()
        
        return x_fft
    

    ########################
    # loss function
    ########################

    def D_loss_inference(self, adv_fake):

#        loss = -torch.log(torch.maximum(1 - adv_fake, torch.tensor(1e-10)))
        loss = nn.MSELoss()(adv_fake, torch.ones_like(adv_fake).cuda() * -1)
        
        return loss


    
    def D_loss_train(self, adv_real, adv_fake):

#        loss = -torch.log(torch.maximum(adv_fake, torch.tensor(1e-10))) - torch.log(torch.maximum(1 - adv_real, torch.tensor(1e-10)))
        loss = nn.MSELoss()(adv_real, torch.ones_like(adv_real).cuda() * -1) + nn.MSELoss()(adv_fake, torch.ones_like(adv_fake).cuda())
        
        return loss




    def robustL1loss(self, x, y):
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2).clamp_min(1e-12)) * torch.sqrt(torch.sum(vy ** 2).clamp_min(1e-12)))
        return torch.log(nn.MSELoss()(x,y) + 1e-5) + torch.log10(1 + 1e-5 - cost)
#        return torch.log(nn.MSELoss()(x, y))



    def robustL1loss_masked(self, x, y, mask):
        xm = torch.fft.ifftn(torch.fft.ifftshift(torch.fft.fftshift(torch.fft.fftn(x)) * mask)).real 
        ym = torch.fft.ifftn(torch.fft.ifftshift(torch.fft.fftshift(torch.fft.fftn(y)) * mask)).real        

        vx = xm - torch.mean(xm)
        vy = ym - torch.mean(ym)

        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2).clamp_min(1e-12)) * torch.sqrt(torch.sum(vy ** 2).clamp_min(1e-12)))
        return torch.log(nn.MSELoss()(xm,ym) + 1e-5) + torch.log10(1 + 1e-5 - cost)
#        return torch.log(nn.MSELoss()(xm,ym))


    def correlationloss(self, x, y):
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2).clamp_min(1e-12)) * torch.sqrt(torch.sum(vy ** 2).clamp_min(1e-12)))
        return -torch.log10(2 - cost)


    ########################
    # IO: images, loss, model
    ########################

    def save_img(self, epoch, idx):
        assembled_images = self.assemble_outputs()
        im = Image.fromarray(assembled_images)
        im = im.convert('L')
        img_filename = '%s/gen_%02d%03d.jpeg' % (self.args.display_dir, epoch, idx)
        im.save(img_filename)



    def assemble_outputs(self):
        row1 = torch.cat((self.flat(self.oA), self.flat(self.A), self.flat(self.oA_rec), self.flat(self.d_A_1), self.flat(self.d_A_2)), 1)
        row2 = torch.cat((self.flat(self.B), self.flat(self.oB), self.flat(self.B_rec) , self.flat(self.d_B_1), self.flat(self.d_B_2)), 1)
       
        large = torch.cat((row1, row2),0).data.cpu().numpy()
#        large[6*32:,:6*32] = smooth(large[6*32:,:6*32], 2.0)
#        large = (large - large.min())/(large.max() - large.min())

        return 255 * large



    def flat(self, x):
        y = torch.zeros((6*32, 6*32), device=x.device)
        x = ((x - x[0].min()) / (torch.maximum(x[0].max() - x[0].min(), torch.tensor(1e-23))))[:,:,::2,::2,::2]
        for i in range(x.size(3)):
            y[i//6*32:(i//6*32+32), i%6*32:(i%6*32+32)] = x[0,0,i]

        return y


        
    def write_loss(self):
        line = np.array(['adv_G_loss', self.adv_G_loss.item(), 'adv_D_loss', self.adv_D_loss.item(),
                        'rec_loss',self.rec_loss.item(), 'ms_loss', self.ms_loss.item()])
                        
        print(line)                

        line = pd.DataFrame(line).T
        line.to_csv(self.args.loss_path, mode='a', header=False) 



    def save_model(self, epoch):
        if epoch % self.args.model_save_freq == 0:
            print('\n--- save the model at epoch %d ---' % (epoch))

            state = {
                    'E': self.E.state_dict(),
                    'G1': self.G1.state_dict(),
                    'G2': self.G2.state_dict(),
                    'dns': self.dns.state_dict(),
                    'disc_add': self.disc_add.state_dict(),
                    'disc_sub': self.disc_sub.state_dict(),
                    'disc_add_spectral': self.disc_add_spectral.state_dict(),
                    'disc_sub_spectral': self.disc_sub_spectral.state_dict(),

                    'E_opt': self.E_opt.state_dict(),
                    'G1_opt': self.G1_opt.state_dict(),
                    'G2_opt': self.G2_opt.state_dict(),
                    'dns_opt': self.dns_opt.state_dict(),
                    'disc_add_opt': self.disc_add_opt.state_dict(),
                    'disc_sub_opt': self.disc_sub_opt.state_dict(),
                    'disc_add_spectral_opt': self.disc_add_spectral_opt.state_dict(),
                    'disc_sub_spectral_opt': self.disc_sub_spectral_opt.state_dict(),

                    'epoch': epoch
                    }

            filename = '%s/%05d.pth' % (self.args.model_dir, epoch)
            torch.save(state, filename)


    
    def load_model(self, model_dir, train=True):
        checkpoint = torch.load(model_dir)
        # weight
        self.E.load_state_dict(checkpoint['E'], False)
        self.G1.load_state_dict(checkpoint['G1'], False)
        self.G2.load_state_dict(checkpoint['G2'], False)
        self.dns.load_state_dict(checkpoint['dns'], False)
        self.disc_add.load_state_dict(checkpoint['disc_add'], False)
        self.disc_sub.load_state_dict(checkpoint['disc_sub'], False)
        self.disc_add_spectral.load_state_dict(checkpoint['disc_add_spectral'], False)
        self.disc_sub_spectral.load_state_dict(checkpoint['disc_sub_spectral'], False)
        # optimizer
        self.E_opt.load_state_dict(checkpoint['E_opt'])
        self.G1_opt.load_state_dict(checkpoint['G1_opt'])
        self.G2_opt.load_state_dict(checkpoint['G2_opt'])
        self.dns_opt.load_state_dict(checkpoint['dns_opt'])
        self.disc_add_opt.load_state_dict(checkpoint['disc_add_opt'])
        self.disc_sub_opt.load_state_dict(checkpoint['disc_sub_opt'])
        self.disc_add_spectral_opt.load_state_dict(checkpoint['disc_add_spectral_opt'])
        self.disc_sub_spectral_opt.load_state_dict(checkpoint['disc_sub_spectral_opt'])

        return checkpoint['epoch']
        
