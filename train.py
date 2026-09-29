#!/usr/bin/env python
# coding: utf-8

import torch
from tqdm import tqdm

from models import get_model
from datasets import get_dataloader

import os, argparse, numpy


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.orthogonal_(m.weight)

    if isinstance(m, torch.nn.Conv3d):
        torch.nn.init.orthogonal_(m.weight)

    if isinstance(m, torch.nn.ConvTranspose3d):
        torch.nn.init.orthogonal_(m.weight)


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    
    resume_path = None
    save_path = 'logs'
    save_img_path = save_path + '/imgs'
    save_model_path = save_path + '/models'
    save_loss_path = save_path + '/loss.csv'
    
    # IO
    parser.add_argument('--dataroot', type=str, default='./data', help='path of data')
    parser.add_argument('--phase', type=str, default='train', help='phase for dataloading')

    parser.add_argument('--display_dir', type=str, default= save_img_path, help='path for saving images')
    parser.add_argument('--model_dir', type=str, default=save_model_path, help='path for saving models')
    parser.add_argument('--loss_path', type=str, default=save_loss_path, help='path for saving losses')
    parser.add_argument('--img_save_freq', type=int, default=1, help='freq (epoch) of saving images')
    parser.add_argument('--model_save_freq', type=int, default=1, help='freq (epoch) of saving models')

    # data
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--num_workers', type=int, default=32, help='# of threads for data loader')
    parser.add_argument('--content_dim', type=int, default=1, help='# of channels')
    
    # network structure
    parser.add_argument('--d_iter', type=int, default=3, help='# of iterations for updating content discriminator')
    parser.add_argument('--dis_scale', type=int, default=3, help='scale of discriminator')
    parser.add_argument('--dis_spectral_norm', action='store_true', help='use spectral normalization in discriminator')

    # training related
    parser.add_argument('--resume', type=str, default=resume_path, help='specified the dir of saved models for resume the training')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--n_epoch', type=int, default=200, help='number of epochs')
    
    
    args = parser.parse_args()
    for path in [args.display_dir, args.model_dir, os.path.dirname(args.loss_path)]:
        if path:
            os.makedirs(path, exist_ok=True)
    train_loader = get_dataloader(args)
    model = get_model(args)
    
    epoch0 = 0
    if args.resume is not None:
        epoch0 = model.load_model(args.resume)
        epoch0 += 1
        print('load from ', args.resume, '\nwith the epoch of', epoch0)



    if args.resume is None:
        model.apply(init_weights)

    lr = args.lr


    # pretrain NEM 
    print('\n--- pretrain NEM ---')
    model.NEM.pretrain(model.E, model.G1, train_loader, epoch_num=0)

    # train
    print('\n--- train model ---')
    global_progress = tqdm(range(epoch0, args.n_epoch), desc=f'Training')
    for epoch in global_progress:
        local_progress=tqdm(train_loader, desc=f'Epoch {epoch}/{args.n_epoch}')
        for _, (ori_images_a, images_b) in enumerate(local_progress):

            ori_images_a = ori_images_a.view(-1,1,64,64,64).cuda().detach()
            images_b = images_b.view(-1,1,64,64,64).cuda().detach()

            # update model
            model.forward(ori_images_a, images_b) 
            local_progress.set_postfix(loss=model.loss.item())    
            
            if _%1000 == 0:
                model.save_img(epoch, _/1000)
            
        # save images for visualization and network
        model.write_loss()
        if epoch%100 == 99:
            model.save_model(epoch)

            lr *= 0.1            

            model.E_opt.param_groups[0]['lr'] = lr
            model.G1_opt.param_groups[0]['lr'] = lr
            model.G2_opt.param_groups[0]['lr'] = lr
            model.dns_opt.param_groups[0]['lr'] = lr
            model.disc_add_opt.param_groups[0]['lr'] = lr
            model.disc_sub_opt.param_groups[0]['lr'] = lr
            model.disc_add_spectral_opt.param_groups[0]['lr'] = lr
            model.disc_sub_spectral_opt.param_groups[0]['lr'] = lr

