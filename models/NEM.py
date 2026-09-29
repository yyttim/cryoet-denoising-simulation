import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
# noise estimation module

from torch.autograd import Variable



def cov(tensor, rowvar=True, bias=False):
    """Estimate a covariance matrix (np.cov)
    https://gist.github.com/ModarTensai/5ab449acba9df1a26c12060240773110
    """
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    return factor * tensor @ tensor.transpose(-1, -2).conj()



class NEM(nn.Module):
    def __init__(self, args, ratio=1):
        super(NEM, self).__init__()
        self.args = args
        self.r = ratio # mixing ratio of densitymap and noise pattern
        
        self.l = nn.L1Loss()
        


    def reparametrize(self, mu):
        z = torch.randn(mu.shape[0], 32, device=mu.device)
        return z

    
    def pretrain(self, E, G, loader, epoch_num=10):
        
        optimizer_E = torch.optim.AdamW(E.parameters(), lr=1e-4)
        optimizer_G = torch.optim.AdamW(G.parameters(), lr=1e-4)
        
        for epoch in range(epoch_num):
            local_progress=tqdm(loader, desc=f'Epoch {epoch}/{epoch_num}')
            for _, (_, x) in enumerate(local_progress):

                x = x.view(-1,1,32,32,32).float().cuda().detach()

                enc = E(x)
                rec_x = G(enc)
                            
                rec_loss = self.robustL1loss(1e4 * x, 1e4 * rec_x) + torch.mean(rec_x).abs()
                enc_mean = torch.mean(enc, dim = 0, keepdim = True)    
                enc_cov = cov(enc.T)
                enc_cov_inv = torch.linalg.inv(enc_cov)
                d = torch.tensor(enc_cov.shape[0])

                kld_loss = 1e-1 * 0.5 * (torch.log(torch.maximum(torch.det(enc_cov), torch.tensor(1e-12))) - torch.log(d) - d + torch.trace(enc_cov_inv) + torch.matmul(torch.matmul(enc_mean, enc_cov_inv), enc_mean.T)) 
                
                loss = rec_loss + kld_loss

                local_progress.set_postfix(loss=loss.item())
                
                optimizer_E.zero_grad()
                optimizer_G.zero_grad()
                loss.backward()
                optimizer_E.step()
                optimizer_G.step()



    def robustL1loss(self, x, y):
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        return 1e-0 * (torch.log10(nn.MSELoss()(x,y)) + torch.log10(1 - cost))    

        
        
    def MAPELoss(self, output, target):
        return torch.mean(torch.abs(target - output)/torch.maximum(torch.abs(target), torch.tensor(1e-12)))



    def fft(self, x):
        x_fft = torch.fft.fftshift(torch.fft.fftn(x))
        x_fft = torch.cat([x_fft.real, x_fft.imag], 1).float()
        
        return x_fft


    def ifft(self, x):
        return torch.fft.ifftn(torch.fft.ifftshift(torch.complex(x[:,0], x[:,1]).view(-1,1,32,32,32))).real      


    
    def forward(self, oA, B, E, G1, G2):
        
#        enc, r = E(self.fft(B)) #get the noise code of fft-B

#        z1 = self.reparametrize(B) # processing of noise code
#        z2 = self.reparametrize(B) # processing of noise code

#        noise_1 = G1(z1, permute = True) # generate noise pattern from noise code in freq domain
#        noise_2 = G1(z2, permute = True) # generate noise pattern from noise code in freq domain

#        noise_1 = noise_1.view(B.shape[0], -1)[:, torch.randperm(noise_1.nelement()//B.shape[0])].view(noise_1.size())
                
        structural_noise_1 = G2(oA)
        structural_noise_2 = G2(oA)
        
        # mode seeking loss to increase diversity
                
#        ms_loss = self.robustL1loss(structural_noise_1, structural_noise_1)
        
        return structural_noise_1, structural_noise_1, structural_noise_2



    def forward_single(self, oA, B, E, G1, G2):
#        enc, r = E(self.fft(B))
        
#        z = self.reparametrize(B)

#        noise = G1(z, permute = True)
        
#        noise = noise.view(B.shape[0], -1)[:, torch.randperm(noise.nelement()//B.shape[0])].view(noise.size())
                
        structural_noise = G2(oA)
        
        out_A_1 = structural_noise 

        return out_A_1, oA



    def forward_single_enc(self, oA, B, E, G1, G2):

#        enc = E(B)
        
#        noise = G1(enc, permute = False)
        
        structural_noise = G2(oA)
        
        out_A_1 = structural_noise 
                
        return out_A_1, out_A_1, structural_noise, structural_noise
    
