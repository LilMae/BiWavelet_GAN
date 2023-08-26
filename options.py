""" Options

This script is largely based on junyanz/pytorch-CycleGAN-and-pix2pix.

Returns:
    [argparse]: Class containing argparse
"""

import argparse
import os
import torch

# pylint: disable=C0103,C0301,R0903,W0622

class Options():
    """Options class

    Returns:
        [argparse]: argparse containing train and test options
    """

    def __init__(self):
        ##
        #
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        ##
        # 데이터 세팅
        self.parser.add_argument('--dataroot', default='', help='path to dataset')
        self.parser.add_argument('--batchsize', type=int, default=2, help='input batch size')
        
        self.parser.add_argument('--z_size', type=int, default=16, help='size of z')
        self.parser.add_argument('--z_ch', type=int, default=1, help='number of channel of z')
        
        self.parser.add_argument('--img_size', type=int, default=128, help='input image size.')
        self.parser.add_argument('--img_ch', type=int, default=3, help='input image size.')
        
        self.parser.add_argument('--feature_size', type=int, default=64, help='input image size.')
        self.parser.add_argument('--feature_ch', type=int, default=64, help='input image size.')
        
        self.parser.add_argument('--signal_size', type=int, default=512, help='length if signal windowing.')
        self.parser.add_argument('--signal_ch', type=int, default=2, help='length if signal windowing.')
        
        self.parser.add_argument('--stride', type=int, default=100, help='stride of signal windowing.')
        self.parser.add_argument('--n_cls', type=int, default=4, help='stride of signal windowing.')
        
        
        # 컴퓨터 리소스
        self.parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
        
        # 네트워크 구성
        
        self.parser.add_argument('--ngf', type=int, default=64)
        self.parser.add_argument('--ndf', type=int, default=64)
        
        self.parser.add_argument('--extralayers', type=int, default=0, help='Number of extra layers on gen and disc')
        self.parser.add_argument('--device', type=str, default='gpu', help='Device: gpu | cpu')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--ngpu', type=int, default=0, help='number of GPUs to use')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
        self.parser.add_argument('--model', type=str, default='ganomaly', help='chooses which model to use. ganomaly')
        
        self.parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')
        self.parser.add_argument('--display', action='store_true', help='Use visdom.')
        self.parser.add_argument('--outf', default='./output', help='folder to output images and model checkpoints')
        self.parser.add_argument('--manualseed', default=-1, type=int, help='manual seed')
        self.parser.add_argument('--proportion', type=float, default=0.1, help='Proportion of anomalies in test set.')
        self.parser.add_argument('--metric', type=str, default='loss', help='Evaluation metric.')

        ##
        # Train
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_image_freq', type=int, default=100, help='frequency of saving real and fake images')
        self.parser.add_argument('--save_test_images', action='store_true', help='Save test images for demo.')
        self.parser.add_argument('--load_weights', action='store_true', help='Load the pretrained weights')
        self.parser.add_argument('--resume', default='', help="path to checkpoints (to continue training)")
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--iter', type=int, default=0, help='Start from iteration i')
        self.parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--w_adv', type=float, default=1, help='Adversarial loss weight')
        self.parser.add_argument('--w_con', type=float, default=50, help='Reconstruction loss weight')
        self.parser.add_argument('--w_enc', type=float, default=1, help='Encoder loss weight.')
        self.parser.add_argument('--conditional', type=bool, default=True, help='Evaluation metric.')
        self.isTrain = True
        self.opt = None

    def parse(self):
        """ Parse Arguments.
        """

        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if self.opt.device == 'gpu':
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        # print('------------ Options -------------')
        # for k, v in sorted(args.items()):
        #     print('%s: %s' % (str(k), str(v)))
        # print('-------------- End ----------------')

        # save to the disk
        if self.opt.name == 'experiment_name':
            self.opt.name = f'BiVi : ver{1}'
        expr_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        test_dir = os.path.join(self.opt.outf, self.opt.name, 'test')

        if not os.path.isdir(expr_dir):
            os.makedirs(expr_dir)
        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)

        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
