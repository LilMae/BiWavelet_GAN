import torch
import torch.nn as nn
import torch.nn.parallel

def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)

class Encoder(nn.Module):
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0, add_final_conv=True):
        """ Feature Extractor

        Args:
            isize (int): input image size (size of x)
            nz (int): size of latent vector (size of z)
            nc (int): input image channel (dim of x)
            ndf (int): latent vector channel (dim of z)
            ngpu (int): number of GPU for parallel  
            n_extra_layers (int, optional): if need, You can insert more layers. Defaults to 0.
            add_final_conv (bool, optional): if True, put Conv kerenl (K x 4 x 4) at last. Defaults to True.
        """
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module('initial-conv-{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial-relu-{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}-{1}-conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cndf),
                            nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}-{1}-relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid-{0}-{1}-conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid-{0}-relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        if add_final_conv:
            main.add_module('final-{0}-{1}-conv'.format(cndf, 1),
                            nn.Conv2d(cndf, nz, 4, 1, 0, bias=False))

        self.main = main

    def forward(self, input):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output


class Decoder(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        """ Feature Extractor

        Args:
            isize (int): input image size (size of x)
            nz (int): size of latent vector (size of z)
            nc (int): input image channel (dim of x)
            ngf (int): same as ndf (dim of z)
            ngpu (int): number of GPU for parallel  
            n_extra_layers (int, optional): if need, You can insert more layers. Defaults to 0.
        """
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module('initial-{0}-{1}-convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial-{0}-batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial-{0}-relu'.format(cngf),
                        nn.ReLU(True))

        csize, _ = 4, cngf
        while csize < isize // 2:
            main.add_module('pyramid-{0}-{1}-convt'.format(cngf, cngf // 2),
                            nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(cngf // 2),
                            nn.BatchNorm2d(cngf // 2))
            main.add_module('pyramid-{0}-relu'.format(cngf // 2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}-{1}-conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{0}-{1}-relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final-{0}-{1}-convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final-{0}-tanh'.format(nc),
                        nn.Tanh())
        self.main = main

    def forward(self, input):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output
    

class NetD(nn.Module):
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0, add_final_conv=True):
        """ Feature Extractor

        Args:
            isize (int): input image size (size of x)
            nz (int): size of latent vector (size of z)
            nc (int): input image channel (dim of x)
            ndf (int): latent vector channel (dim of z)
            ngpu (int): number of GPU for parallel  
            n_extra_layers (int, optional): if need, You can insert more layers. Defaults to 0.
            add_final_conv (bool, optional): if True, put Conv kerenl (K x 4 x 4) at last. Defaults to True.
        """
        super(NetD, self).__init__()
        model = Encoder(isize, nz, nc, ndf, ngpu, n_extra_layers, add_final_conv)
        layers = list(model.main.children())

        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        features = self.features(x)
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)

        return classifier
    
if __name__ == '__main__':
    netG = Decoder(isize=256,
                    nz=100,
                    nc=1,
                    ngf=64,
                    ngpu=1)
    
    netD = NetD(isize=256,
                nz=100,
                nc=1,
                ndf=64,
                ngpu=1)
    
    Enc = Encoder(isize=256,
                nz=100,
                nc=1,
                ndf=64,
                ngpu=1)
    