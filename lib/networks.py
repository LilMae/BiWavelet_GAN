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
    """
    DCGAN ENCODER NETWORK
    """

    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0, add_final_conv=True):
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

##
class Decoder(nn.Module):
    """
    DCGAN DECODER NETWORK
    """
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
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


class NetG(nn.Module):
    """Generator for generating signal and xwt image

        input
            z : batch_size x 1 x nz
            x_style : batch_size x 1 x ns
        output : 
            signal : batch_size x 2 x window_size
            xwt : batch_size x nc x im_size
    """
    def __init__(self, opt):
        super(NetG, self).__init__()
        # Step 1 : z와 x의 style을 입력받아 middle_feature을 생성
        self.middle_feature = Decoder(
            isize   = opt.middle_size,      # 출력 크기는 middle_size
            nz      = opt.nz+opt.ns,        # 입력 크기는 x_feature과 z의 cat
            nc      = opt.middle_ch,        # 출력 채널은 middle_ch
            ngf     = opt.ngf,
            ngpu    = opt.ngpu 
        )
        
        # Step 2 : middle_feture 을 입력받아 feature 생성
        self.feature = Encoder(
            isize   = opt.middle_size,     # 입력 크기는 middle_size
            nz      = opt.nz,              # 출력 크기는 nz
            nc      = opt.middle_ch,       # 입력 채널은 middle_ch
            ndf     = opt.ngf,
            ngpu    = opt.ngpu
        )
        
        
        # Step 3 : feature를 입력 받아 signal 생성
        self.decoder_sig = Decoder(
            isize = opt.window_size, 
            nz    = opt.nz,
            nc    = 2,
            ngf   = opt.ngf,
            ngpu  = opt.ngpu
        )
        # Step 4 : feature를 입력받아 xwt 생성
        self.decoder_xwt = Decoder(
            isize = opt.im_size, 
            nz    = opt.nz,
            nc    = opt.nc,
            ngf   = opt.ngf,
            ngpu  = opt.ngpu
        )
        
    def forward(self, z, x_style):
        
        input_data = torch.cat(z, x_style)
        middle_feature = self.middle_feature(input_data)
        feature = self.feature(middle_feature)
        
        signal = self.decoder_sig(feature)
        
        xwt = self.decoder_xwt(feature)
        
        return signal, xwt

class NetE(nn.Module):
    """Encoder for generating feature

        input
            signal : batch_size x 2 x window_size
            xwt : batch_size x nc x im_size
            
        output
            feature : batch_size x 1 x nz
    """
    def __init__(self, opt):
        super(NetE, self).__init__()
        # Step 1 : Signal 을 입력받아 middle feature 생성
        self.encoder_sig = Encoder(
            isize   = opt.window_size,     # 입력 크기는 window_size
            nz      = opt.nz,              # 출력 크기는 nz
            nc      = 2,                   # 입력 채널은 2
            ndf     = opt.ngf,
            ngpu    = opt.ngpu
        )
        self.middle_sig = Decoder(
            isize   = opt.middle_size,      # 출력 크기는 middle_size
            nz      = opt.nz+opt.ns,        # 입력 크기는 nz
            nc      = opt.middle_ch,        # 출력 채널은 middle_ch
            ngf     = opt.ngf,
            ngpu    = opt.ngpu 
        )
        
        # Step 2 : xwt 를 입력받아 middle feature 생성
        self.encoder_xwt = Encoder(
            isize   = opt.im_size,         # 입력 크기는 im_size
            nz      = opt.nz,              # 출력 크기는 nz
            nc      = opt.nc,              # 입력 채널은 nc
            ndf     = opt.ngf,
            ngpu    = opt.ngpu
        )
        self.middle_xwt = Decoder(
            isize   = opt.middle_size,      # 출력 크기는 middle_size
            nz      = opt.nz+opt.ns,        # 입력 크기는 nz
            nc      = opt.middle_ch,        # 출력 채널은 middle_ch
            ngf     = opt.ngf,
            ngpu    = opt.ngpu 
        )
        
        # Step 3 : middle_feature 두개를 입력받아 feature 생성
        self.feature = Encoder(
            isize   = opt.middle_size,      # 입력 크기는 middle_size
            nz      = opt.nz,               # 출력 크기는 nz
            nc      = opt.middle_ch,        # 입력 채널은 middle_ch
            ndf     = opt.ngf,
            ngpu    = opt.ngpu
        )
        
    def forward(self, signal, xwt):
        middle_sig = self.encoder_sig(signal)
        middle_sig = self.middle_sig(middle_sig)
        
        middle_xwt = self.encoder_xwt(xwt)
        middle_xwt = self.middle_xwt(middle_xwt)
        
        middle_feature = torch.cat(middle_sig, middle_xwt)
        feature = self.feature(middle_feature)
    
        return feature
    
class NetD(nn.Module):
    
    def __init__(self, opt):
        super(NetD, self).__init__()
        # Step 1 : Signal 을 입력받아 sig_feature 생성
        self.encoder_sig = Encoder(
            isize   = opt.window_size,     # 입력 크기는 window_size
            nz      = opt.nz,              # 출력 크기는 nz
            nc      = 2,                   # 입력 채널은 2
            ndf     = opt.ndf,
            ngpu    = opt.ngpu
        )

        
        # Step 3 : feature 두개를 입력받아 예측 생성
        model = Encoder(
            isize   = opt.nz,               # 입력 크기는 nz
            nz      = 1,               # 출력 크기는 1
            nc      = 2,                    # 입력 채널은 2
            ndf     = opt.ndf,
            ngpu    = opt.ngpu
        )
        layers = list(model.main.children())
        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module('Sigmoid', nn.Sigmoid())
        
    def forward(self, signal, feature):
        sig_feature = self.encoder_sig(signal)
        feature = torch.cat(sig_feature, feature)
        feature = self.features(feature)
        classifier = self.classifier(feature)
        classifier = classifier.view(-1, 1).squeeze(1)
        
        return classifier

# class NetD(nn.Module):
#     def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0, add_final_conv=True):
#         """ Feature Extractor

#         Args:
#             isize (int): input image size (size of x)
#             nz (int): size of latent vector (size of z)
#             nc (int): input image channel (dim of x)
#             ndf (int): latent vector channel (dim of z)
#             ngpu (int): number of GPU for parallel  
#             n_extra_layers (int, optional): if need, You can insert more layers. Defaults to 0.
#             add_final_conv (bool, optional): if True, put Conv kerenl (K x 4 x 4) at last. Defaults to True.
#         """
#         super(NetD, self).__init__()
        
        
#         # Enc X : (nc, isize, isize) -> (nz, 1, 1) -> (nz, 16, 16)
#         self.Enc_x1 = Encoder(isize = isize,  
#                         nz = nz,     
#                         nc = nc,        
#                         ndf = ndf,    
#                         ngpu = ngpu, 
#                         n_extra_layers=0, 
#                         add_final_conv=True)
#         self.Enc_x2 = Decoder(isize=16,
#                               nz = nz,
#                               nc=nz,
#                               ngf=ndf,
#                               ngpu=ngpu,
#                               n_extra_layers=0
#                               )

#         # Enc z : (nz, 1, 1) -> (nz, 16, 16)
#         self.Enc_z = Decoder(isize = 16,     
#                         nz = nz,     
#                         nc = nz,       
#                         ngf = ndf,    
#                         ngpu = ngpu, 
#                         n_extra_layers=0)
        

#         self.model = Encoder(isize = 16,  
#                         nz = 1,         
#                         nc = nz*3,      
#                         ndf = ndf,      
#                         ngpu = ngpu, 
#                         n_extra_layers=n_extra_layers, 
#                         add_final_conv=True)
        
#         # Enc sig : (2, isize) -> (nz, 1, 1)
#         self.Enc_sig1 = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(in_features=2*isize,
#                         out_features=nz),
#             nn.Linear(in_features=nz,
#                         out_features=nz)
#         )
#         # feature(nz, 1, 1) ->img(nc, isize, isize)
#         self.Enc_sig2 = Decoder(isize = 16,  
#                         nz = nz,     
#                         nc = nc,        
#                         ngf = ndf,    
#                         ngpu = ngpu, 
#                         n_extra_layers=0)


        
#         layers = list(self.model.main.children())
#         self.classifier = nn.Sequential(layers[-1])
#         self.model.add_module('Sigmoid', nn.Sigmoid())

#     def forward(self, x, z, sig):
#         x = self.Enc_x1(x)
#         x_feateure = self.Enc_x2(x)
        
#         # print(f'x_feature : {x_feateure.size()}')
        
#         z_feature = self.Enc_z(z)
        
#         # print(f'z_feature : {z_feature.size()}')
        
#         sig_feature = self.Enc_sig1(sig)
#         sig_feature = sig_feature.unsqueeze(dim=-1).unsqueeze(dim=-1)
#         sig_feature = self.Enc_sig2(sig_feature)

#         features = torch.cat((x_feateure,z_feature, sig_feature), dim=1)
#         features = self.model(features)
#         # classifier = self.classifier(features)
#         classifier = features.view(-1, 1).squeeze(1)

#         return classifier
    

# class NetG(nn.Module):
#     """
#     img -> sig
#     """
#     def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0, add_final_conv=True):
#         super(NetG, self).__init__()
#         self.isize = isize
#         # img'(nc, isize, isize) -> feature(nz, 1, 1)
#         self.encoder =Encoder(isize = isize,  
#                         nz = nz,     
#                         nc = nc,        
#                         ndf = ngf,    
#                         ngpu = ngpu, 
#                         n_extra_layers=0, 
#                         add_final_conv=True)
#         # feature(nz, 1, 1) -> sig'(2, isize)
#         self.fc = nn.Sequential(
#             nn.Linear(in_features=nz,
#                         out_features=2*isize),
#             nn.Linear(in_features=2*isize,
#                         out_features=2*isize)
#         )

#     def forward(self, x):
#         feature = self.encoder(x)
#         feature = feature.sqeeze()
#         sig = self.fc(feature)

#         sig1 = sig[:,:,:self.isize]
#         sig2 = sig[:,:,self.isize:]

#         sig = torch.concat((sig1,sig2),dim=1)

#         return sig

# class NetE(nn.Module):
#     """
#     sig -> img
#     """
#     def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0, add_final_conv=True):
#         super(NetE, self).__init__()

#         # sig(2,isize) -> feature(nz, 1, 1)
#         self.fc = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(in_features=2*isize,
#                         out_features=nz),
#             nn.Linear(in_features=nz,
#                         out_features=nz)
#         )
#         # feature(nz, 1, 1) ->img(nc, isize, isize)
#         self.decoder = Decoder(isize = isize,  
#                         nz = nz,     
#                         nc = nc,        
#                         ngf = ndf,    
#                         ngpu = ngpu, 
#                         n_extra_layers=0)

#     def forward(self, x):

#         feature = self.fc(x)
#         feature = feature.unsqueeze(dim=-1).unsqueeze(dim=-1)
#         img = self.decoder(feature)
        
#         return img

# if __name__ == '__main__':
    

#     model = NetD(isize=256, nz=64, nc=1, ndf=16, ngpu=1)
#     z = torch.randn(size=(2,64,1,1))
#     x = torch.randn(size=(2,1,256,256))
#     res = model(x,z)
    
#     print(res.size())