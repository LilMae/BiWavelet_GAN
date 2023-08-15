from collections import OrderedDict
import os
import time
import numpy as np
import torch
from tqdm import tqdm

from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

from lib.networks import Encoder, Decoder, NetD, weights_init, NetG, NetE
from lib.visualizer import Visualizer, save_current_images
from lib.loss import l2_loss
from lib.evaluate import evaluate
from lib.model import BaseModel

class BiVi(BaseModel):

    @property
    def name(self): return 'BiVi'

    def __init__(self, opt, dataloader):
        super(BiVi, self).__init__(opt, dataloader)

        self.epoch = 0
        self.times = []
        self.total_steps = 0

        # Generator1
        # z(nz, 1, 1) -> img'(nc, isize, isize)
        self.net_g1 = Decoder(isize=self.opt.isize,
                            nz = self.opt.nz,
                            nc = self.opt.nc,
                            ngf=self.opt.ngf,
                            ngpu=self.opt.ngpu
                            ).to(self.device)
        
        # Generator2
        # img'(nc, isize, isize) -> sig'(2, isize)
        self.net_g2 = NetG(isize=self.opt.isize,
                            nz = self.opt.nz,
                            nc = self.opt.nc,
                            ngf=self.opt.ngf,
                            ngpu=self.opt.ngpu
                            ).to(self.device)
        
        # Encoder1
        # img'(nc, isize, isize) -> z(nz, 1, 1)
        if self.opt.conditional:
            self.net_e1 = Encoder(isize = self.opt.isize,  
                        nz = self.opt.nz-4,     
                        nc = self.opt.nc,        
                        ndf = self.opt.ndf,    
                        ngpu = self.opt.ngpu,
                        ).to(self.device)
        else:
            self.net_e1 = Encoder(isize = self.opt.isize,  
                            nz = self.opt.nz,     
                            nc = self.opt.nc,        
                            ndf = self.opt.ndf,    
                            ngpu = self.opt.ngpu,
                            ).to(self.device)
        
        #Encoder2
        # sig(2, isize) -> img'(nc, isize, isize) 
        self.net_e2 = NetE(isize=self.opt.isize,
                            nz = self.opt.nz,
                            nc = self.opt.nc,
                            ndf=self.opt.ngf,
                            ngpu=self.opt.ngpu
                            ).to(self.device)
        
        # Discriminator 
        # (img, sig, z) -> Real/False
        self.netd = NetD(isize = self.opt.isize, 
                         nz = self.opt.nz, 
                         nc = self.opt.nc, 
                         ndf  = self.opt.ndf,
                         ngpu = self.opt.ngpu
                         ).to(self.device)
        
        # Loss
        self.l_bce = nn.CrossEntropyLoss()
        self.l_l1 = nn.L1Loss()

        # Optimizer
        if self.opt.isTrain:
            self.net_g1.train()
            self.net_g2.train()
            self.net_e1.train()
            self.net_e2.train()
            self.netd.train()

            self.optimizer_ge    = optim.Adam(
                list(self.net_g1.parameters())+list(self.net_g2.parameters()) + 
                list(self.net_e1.parameters())+list(self.net_e2.parameters()),
                lr=self.opt.lr, betas=(self.opt.beta1, 0.999)
            )
            self.optimizer_d   = optim.Adam(
                self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999)
            )

    def reinit_d(self):
        self.netd.apply(weights_init)
    
    def optimize_params(self, z, img, sig, one_hot):
        self.optimizer_d.zero_grad()
        self.optimizer_ge.zero_grad()

        g_img = self.net_g1(z)
        g_sig = self.net_g2(g_img)

        e_img = self.net_e2(sig)
        if self.opt.conditional:
            e_z = self.net_e1(e_img)
            e_z = torch.cat((e_z, one_hot), 1)
        else:
            e_z = self.net_e1(e_img)
        
        out_real = self.netd(x  = e_img,
                             z  = e_z,
                             sig= sig)
        out_fake = self.netd(x  = g_img,
                             z  = z,
                             sig= g_sig)
        
        real_label = torch.ones(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        fake_label = torch.zeros(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)

        loss_d = self.l_bce(self.netd(z=z.detach(), x=g_img.detach(), sig=g_sig.detach()), fake_label) + \
                self.l_bce(self.netd(z=e_z.detach(), x=e_img.detach(), sig=sig.detach()), real_label)
        loss_d.backward()
        self.optimizer_d.step()

        loss_generation = self.l_bce(out_real, fake_label) + self.l_bce(out_fake, real_label)
        loss_convert = self.l_l1(self.net_g2(img), sig) + self.l_l1(self.net_e2(sig), img)
        loss_ge = loss_generation+loss_convert
        loss_ge.backward(retain_graph=True)
        self.optimizer_ge.step()

        return loss_ge.item(), loss_d.item()

    def train_one_epoch(self):

        self.net_g1.train()
        self.net_g2.train()
        self.net_e1.train()
        self.net_e2.train()
        self.netd.train()

        epoch_iter = 0

        temp_loss_ge = 0.0
        temp_loss_d = 0.0

        for img, signal, state in tqdm(self.dataloader['train']):

            self.total_steps += self.opt.batchsize
            epoch_iter += self.opt.batchsize

            if self.opt.conditional:
                one_hot = torch.nn.functional.one_hot(state, num_classes=4)
                one_hot = one_hot.unsqueeze(dim=-1)
                one_hot = one_hot.unsqueeze(dim=-1)
                one_hot = one_hot.to(self.device)
            
                z = torch.randn(size=(self.opt.batchsize, (self.opt.nz-4), 1, 1)).to(self.device).float()
                z = torch.cat((z, one_hot), 1)
            
            else:
                z = torch.randn(size=(self.opt.batchsize, self.opt.nz, 1, 1)).to(self.device).float()
                one_hot = False
        
            img = img.to(self.device).float()

            loss_ge, loss_d = self.optimize_params(z=z,
                                 img=img,
                                 sig=signal,
                                 one_hot=one_hot)
            
            temp_loss_ge += loss_ge
            temp_loss_d += loss_d
        
        self.loss_ge = temp_loss_ge/len(self.dataloader['train'])
        self.loss_d = temp_loss_d/len(self.dataloader['train'])


    def train(self):
         
         self.total_steps = 0

         print('Start Trainning')
         with open('log.txt', mode='w')as f:
             f.write('Start Trainning......')

         for self.epoch in range(self.opt.iter, self.opt.niter):
    
            self.train_one_epoch()
            with open('log.txt', mode='a') as f:
                f.write(f'Epoch : {self.epoch}')
                f.write(f'loss_ge : {self.loss_ge}')
                f.write(f'loss_d : {self.loss_d}')

            if self.epoch%10 == 0:
                self.save_weights(self.epoch)