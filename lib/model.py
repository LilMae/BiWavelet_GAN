"""GANomaly
"""
# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915
##CUDA_LAUNCH_BLOCKING=1
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

from lib.networks import Encoder, Decoder, NetD, weights_init
from lib.visualizer import Visualizer, save_current_images
from lib.loss import l2_loss
from lib.evaluate import evaluate



import matplotlib.pyplot as plt


class BaseModel():
    """ Base Model for ganomaly
    """
    def __init__(self, opt, dataloader):
        ##
        # Seed for deterministic behavior
        self.seed(opt.manualseed)

        # Initalize variables.
        self.opt = opt
        self.visualizer = Visualizer(opt)
        self.dataloader = dataloader
        self.trn_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        self.tst_dir = os.path.join(self.opt.outf, self.opt.name, 'test')
        self.device = torch.device("cuda:0" if self.opt.device != 'cpu' else "cpu")
        torch.autograd.set_detect_anomaly(True)

    ##
    def seed(self, seed_value):
        """ Seed 
        
        Arguments:
            seed_value {int} -- [description]
        """
        # Check if seed is default value
        if seed_value == -1:
            return

        # Otherwise seed all functionality
        import random
        random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        np.random.seed(seed_value)
        torch.backends.cudnn.deterministic = False

    ##
    def get_errors(self):
        """ Get netD and netG errors.

        Returns:
            [OrderedDict]: Dictionary containing errors.
        """

        errors = OrderedDict([
            ('err_d', self.err_d.item()),
            ('err_g', self.err_g.item()),
            ('err_g_adv', self.err_g_adv.item()),
            ('err_g_con', self.err_g_con.item()),
            ('err_g_enc', self.err_g_enc.item())])

        return errors

    ##
    def get_current_images(self):
        """ Returns current images.

        Returns:
            [reals, fakes, fixed]
        """

        reals = self.x.data
        fakes = self.Gz.data

        return reals, fakes

    ##
    def save_weights(self, epoch):
        """Save netG and netD weights for the current epoch.

        Args:
            epoch ([int]): Current epoch number.
        """

        weight_dir = os.path.join(self.opt.outf, self.opt.name, 'train', 'weights')
        if not os.path.exists(weight_dir): os.makedirs(weight_dir)

        torch.save({'epoch': epoch + 1, 'state_dict': self.netg.state_dict()},
                   f'%s/netG_{epoch}.pth' % (weight_dir))
        torch.save({'epoch': epoch + 1, 'state_dict': self.netd.state_dict()},
                   f'%s/netD_{epoch}.pth' % (weight_dir))
        torch.save({'epoch': epoch + 1, 'state_dict': self.nete.state_dict()},
                   f'%s/netE_{epoch}.pth' % (weight_dir))

    ##
    def train_one_epoch(self):
        """ Train the model for one epoch.
        """

        self.netg.train()
        epoch_iter = 0
        for data in tqdm(self.dataloader['train'], leave=False, total=len(self.dataloader['train'])):
            self.total_steps += self.opt.batchsize
            epoch_iter += self.opt.batchsize

            self.set_input(data)
            # self.optimize()
            self.optimize_params()

            if self.total_steps % self.opt.print_freq == 0:
                errors = self.get_errors()
                if self.opt.display:
                    counter_ratio = float(epoch_iter) / len(self.dataloader['train'].dataset)
                    self.visualizer.plot_current_errors(self.epoch, counter_ratio, errors)


        print(">> Training model %s. Epoch %d/%d" % (self.name, self.epoch+1, self.opt.niter))
        print(f'Errs : {errors}')
        # self.visualizer.print_current_errors(self.epoch, errors)

    ##
    def train(self):
        """ Train the model
        """

        ##
        # TRAIN
        self.total_steps = 0
        best_auc = 0

        # Train for niter epochs.
        print(">> Training model %s." % self.name)
        for self.epoch in range(self.opt.iter, self.opt.niter):
            # Train for one epoch
            self.train_one_epoch()
            res = self.test()
            if res[self.opt.metric] > best_auc:
                best_auc = res[self.opt.metric]
                self.save_weights(self.epoch)
            self.visualizer.print_current_performance(res, best_auc)
        print(">> Training model %s.[Done]" % self.name)

    ##
    def test(self):
        """ Test GANomaly model.

        Args:
            dataloader ([type]): Dataloader for the test set

        Raises:
            IOError: Model weights not found.
        """
        with torch.no_grad():
            # Load the weights of netg and netd.
            if self.opt.load_weights:
                path = "./output/{}/{}/train/weights/netG.pth".format(self.name.lower(), self.opt.dataset)
                pretrained_dict = torch.load(path)['state_dict']

                try:
                    self.netg.load_state_dict(pretrained_dict)
                except IOError:
                    raise IOError("netG weights not found")
                print('   Loaded weights.')

            self.opt.phase = 'test'

            # Create big error tensor for the test set.
            self.an_scores = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.long,    device=self.device)
            self.latent_i  = torch.zeros(size=(len(self.dataloader['test'].dataset), self.opt.nz), dtype=torch.float32, device=self.device)
            self.latent_o  = torch.zeros(size=(len(self.dataloader['test'].dataset), self.opt.nz), dtype=torch.float32, device=self.device)

            # print("   Testing model %s." % self.name)
            self.times = []
            self.total_steps = 0
            epoch_iter = 0
            for i, data in enumerate(self.dataloader['test'], 0):
                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()
                self.set_input(data)
                self.fake, latent_i, latent_o = self.netg(self.input)

                error = torch.mean(torch.pow((latent_i-latent_o), 2), dim=1)
                time_o = time.time()

                self.an_scores[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = error.reshape(error.size(0))
                self.gt_labels[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = self.gt.reshape(error.size(0))
                self.latent_i [i*self.opt.batchsize : i*self.opt.batchsize+error.size(0), :] = latent_i.reshape(error.size(0), self.opt.nz)
                self.latent_o [i*self.opt.batchsize : i*self.opt.batchsize+error.size(0), :] = latent_o.reshape(error.size(0), self.opt.nz)

                self.times.append(time_o - time_i)

                # Save test images.
                if self.opt.save_test_images:
                    dst = os.path.join(self.opt.outf, self.opt.name, 'test', 'images')
                    if not os.path.isdir(dst):
                        os.makedirs(dst)
                    real, fake, _ = self.get_current_images()
                    vutils.save_image(real, '%s/real_%03d.eps' % (dst, i+1), normalize=True)
                    vutils.save_image(fake, '%s/fake_%03d.eps' % (dst, i+1), normalize=True)

            # Measure inference time.
            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)

            # Scale error vector between [0, 1]
            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (torch.max(self.an_scores) - torch.min(self.an_scores))
            # auc, eer = roc(self.gt_labels, self.an_scores)
            auc = evaluate(self.gt_labels, self.an_scores, metric=self.opt.metric)
            performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), (self.opt.metric, auc)])

            if self.opt.display_id > 0 and self.opt.phase == 'test':
                counter_ratio = float(epoch_iter) / len(self.dataloader['test'].dataset)
                self.visualizer.plot_performance(self.epoch, counter_ratio, performance)
            return performance

##
class BiGAN(BaseModel):

    @property
    def name(self): return 'BiGAN'

    def __init__(self, opt, dataloader):
        super(BiGAN, self).__init__(opt, dataloader)

        self.epoch = 0
        self.times = []
        self.total_steps = 0

        """
        
        BiGAN에는 Encoder, Generator, Decoder가 존재함
        
        """
        print(f'Building Generator ...')
        self.netg = Decoder(isize=self.opt.isize,
                            nz=self.opt.nz,
                            nc=self.opt.nc,
                            ngf=self.opt.ngf,
                            ngpu=self.opt.ngpu
                            ).to(self.device)
        print(f'Building Discriminator ...')
        self.netd = NetD(isize = self.opt.isize, 
                         nz = self.opt.nz, 
                         nc = self.opt.nc, 
                         ndf  = self.opt.ndf,
                         ngpu = self.opt.ngpu
                         ).to(self.device)
        print(f'Building Encoder ... ')

        if self.opt.conditional:
            self.nete = Encoder(isize = self.opt.isize,  
                        nz = self.opt.nz-4,     
                        nc = self.opt.nc,        
                        ndf = self.opt.ndf,    
                        ngpu = self.opt.ngpu,
                        ).to(self.device)
        else:
            self.nete = Encoder(isize = self.opt.isize,  
                            nz = self.opt.nz,     
                            nc = self.opt.nc,        
                            ndf = self.opt.ndf,    
                            ngpu = self.opt.ngpu,
                            ).to(self.device)
        
        self.netg.apply(weights_init)
        self.netd.apply(weights_init)
        self.nete.apply(weights_init)

        ##
        if self.opt.resume != '':
            print("\nLoading pre-trained networks.")
            self.opt.iter = torch.load(os.path.join(self.opt.resume, 'netG.pth'))['epoch']
            self.netg.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netG.pth'))['state_dict'])
            self.netd.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netD.pth'))['state_dict'])
            self.nete.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netE.pth'))['state_dict'])
            print("\tDone.\n")

        # 사용하는 손실함수들
        self.l_bce = nn.CrossEntropyLoss()
        
        # Setup optimizer
        if self.opt.isTrain:
            self.netg.train()
            self.netd.train()
            self.nete.train()
            self.optimizer_d = optim.Adam(self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_ge = optim.Adam(list(self.netg.parameters()) +
                                           list(self.nete.parameters()), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

    ##
    def reinit_d(self):
        """ Re-initialize the weights of netD
        """
        self.netd.apply(weights_init)
        print('   Reloading net d')

    def train(self):
        
        self.total_steps = 0
        best_auc = 0
        
        log_file = open('log.txt', mode='w')
        
        # Train for niter epochs.
        print(">> Training model %s." % self.name)
        for self.epoch in range(self.opt.iter, self.opt.niter):
            log_file.write(f'epoch : {self.epoch} \n')
            # Train for one epoch
            self.train_one_epoch()
            log_file.write(f'err_ge : {self.loss_ge} \n')
            log_file.write(f'err_d : {self.loss_d} \n')
            
            # loss_img, loss_compress = self.test()
            # print(f'loss for img : {loss_img}')
            # print(f'loss_for compress : {loss_compress}')
            # log_file.write(f'loss for img : {loss_img}')
            # log_file.write(f'loss_for compress : {loss_compress}')
            
            # 매 10에폭마다 저장
            if self.epoch%10 == 0:
                self.save_weights(self.epoch)
                save_current_images(epoch=self.epoch, reals=self.x, fakes=self.Gz, dir=self.opt.outf)


            
            
        print(">> Training model %s.[Done]" % self.name)        
        log_file.close()
    
    def optimize_params(self, z, x, one_hot):
        self.x = x
        self.optimizer_ge.zero_grad()
        self.optimizer_d.zero_grad()
        # print('Start Forwarding')
        #Generator - Forward
        # print(f'self.z : {self.z.size()}')


        Gz = self.netg(z)
        self.Gz = Gz
        # print(f'self.Gz : {self.Gz.size()}')
        
        #Encoder - Forward
        if self.opt.conditional:
            Ex = self.nete(x)
            Ex = torch.cat((Ex, one_hot), 1)
        else:
            Ex = self.nete(x)
        #Discriminator - Forward
        
        # print(f'Ex : {self.Ex.size()}')
        
        out_real = self.netd(x, Ex)
        out_fake = self.netd(Gz.detach(), z.detach())
        
        # print(f'Forawrd Finish')

        real_label = torch.ones(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        fake_label = torch.zeros(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        
        loss_ge = self.l_bce(out_real, fake_label) + self.l_bce(out_fake, real_label)

        loss_ge.backward(retain_graph=True)
        self.optimizer_ge.step()
        loss_d = self.l_bce(self.netd(x.detach(), Ex.detach()), real_label) + self.l_bce(out_fake, fake_label)
        
        loss_d.backward()
        self.optimizer_d.step()
        
        self.loss_ge = loss_ge.item()
        self.loss_d = loss_d.item()
    
        
        
    def get_errors(self):
        """ Get netD and netG errors.

        Returns:
            [OrderedDict]: Dictionary containing errors.
        """

        errors = OrderedDict([
            ('loss_d', self.loss_d),
            ('loss_ge', self.loss_ge)])

        return errors
    
    def train_one_epoch(self):
        
        self.netg.train()
        self.netd.train()
        self.nete.train()
        
        epoch_iter = 0
        for img, signal, state in tqdm(self.dataloader['train']):
            
            # plt.imsave(f'testing-{epoch_iter}.jpg',img[0][0])
            
            
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
            
            x = img.to(self.device).float()
            
            self.optimize_params(z, x, one_hot)
            
            if self.total_steps % self.opt.print_freq == 0:
                errors = self.get_errors()
                if self.opt.display:
                    counter_ratio = float(epoch_iter) / len(self.dataloader['train'].dataset)
                    self.visualizer.plot_current_errors(self.epoch, counter_ratio, errors)

            if self.total_steps % self.opt.save_image_freq == 0:
                reals, fakes = self.get_current_images()
                self.visualizer.save_current_images(self.epoch, reals, fakes)
                if self.opt.display:
                    self.visualizer.display_current_images(reals, fakes)
            
    def test(self):
        
        with torch.no_grad():
            
            loss_img = 0.0
            loss_compress = 0.0
            
            for img, signal, state in tqdm(self.dataloader['test']):
                
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
                
                x = img.to(self.device).float()
                
                Gz = self.netg(z)
                
                if self.opt.conditional:
                    Ex = self.nete(x)
                    Ex = torch.cat((Ex, one_hot), 1)
                else:
                    Ex = self.nete(x)
                
                loss_img += torch.mean(torch.pow((x-Gz), 2), dim=1)
                loss_compress += torch.mean(torch.pow((z-Ex), 2), dim=1)
        
        return loss_img.mean(), loss_compress.mean()