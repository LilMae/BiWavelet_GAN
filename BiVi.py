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

        self.Generator = NetG(opt=opt)
        self.Encoder = NetE(opt=opt)
        self.Discriminator = NetD(opt=opt)
        
        # Loss
        self.l_bce = nn.CrossEntropyLoss()
        self.l_l1 = nn.L1Loss()

        # Optimizer
        if self.opt.isTrain:
            self.Generator.train()
            self.Encoder.train()
            self.Discriminator.train()

            self.optimizer_ge    = optim.Adam(
                list(self.Generator.parameters())+list(self.Encoder.parameters()),
                lr=self.opt.lr, betas=(self.opt.beta1, 0.999)
            )
            self.optimizer_d   = optim.Adam(
                self.Discriminator.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999)
            )

    def reinit_d(self):
        self.Discriminator.apply(weights_init)
    
    def optimize_params(self, z, img, sig, one_hot):
        self.optimizer_d.zero_grad()
        self.optimizer_ge.zero_grad()

        fake_sig, fake_xwt, fake_feature = self.Generator(z, sig)
        real_feature = self.Encoder(sig, img)
        
        
        out_real = self.Discriminator(
            signal = fake_sig,
            feature = fake_feature
        )
        out_fake = self.Discriminator(
            signal = sig,
            feature = real_feature
        )
        
        real_label = torch.ones(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        fake_label = torch.zeros(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)

        loss_d = self.l_bce(self.Discriminator(fake_sig.detach(), fake_feature.detach()), fake_label) + \
                self.l_bce(self.Discriminator(sig.detach(), real_feature.detach()), real_label)
        loss_d.backward()
        self.optimizer_d.step()

        loss_generation = self.l_bce(out_real, fake_label) + self.l_bce(out_fake, real_label)
        loss_convert = self.l_l1(fake_sig, sig) + self.l_l1(fake_xwt, img)
        loss_ge = loss_generation+loss_convert
        loss_ge.backward(retain_graph=True)
        self.optimizer_ge.step()

        return loss_ge.item(), loss_d.item()

    def train_one_epoch(self):

        self.Generator.train()
        self.Encoder.train()
        self.Discriminator.train()

        epoch_iter = 0

        temp_loss_ge = 0.0
        temp_loss_d = 0.0

        for batch in tqdm(self.dataloader['train']):
            
            state = batch['class']
            img = batch['Wcoh']
            signal = batch['sensor']

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