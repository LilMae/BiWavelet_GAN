import torch
import torch.nn as nn
import torch.fft
from options import Options
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
        
    def forward(self, z, x):
        
        x_style = top3_frequencies(x[0])
        
        input_data = torch.cat(z, x_style)
        middle_feature = self.middle_feature(input_data)
        feature = self.feature(middle_feature)
        
        signal = self.decoder_sig(feature)
        
        xwt = self.decoder_xwt(feature)
        
        return signal, xwt, feature

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

def extract_and_reconstruct(tensor):
    # 입력 텐서의 크기 확인
    
    batch, ch,  n = tensor.shape
    
    # FFT 수행
    fft_result = torch.fft.fft(tensor, n=n)
    
    # 3rd Order까지의 동기 성분만 유지
    fft_result[:, 4:] = 0
    fft_result[:, 1:4] = 0
    
    # IFFT 수행하여 원래의 시간 영역으로 복구
    reconstructed = torch.fft.ifft(fft_result, n=n).real

    return reconstructed

    
class Encoder_1D(nn.Module):
    """
    DCGAN ENCODER NETWORK
    """

    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0, add_final_conv=True):
        super(Encoder_1D, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module('initial-conv-{0}-{1}'.format(nc, ndf),
                        nn.Conv1d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial-relu-{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}-{1}-conv'.format(t, cndf),
                            nn.Conv1d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cndf),
                            nn.BatchNorm1d(cndf))
            main.add_module('extra-layers-{0}-{1}-relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid-{0}-{1}-conv'.format(in_feat, out_feat),
                            nn.Conv1d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(out_feat),
                            nn.BatchNorm1d(out_feat))
            main.add_module('pyramid-{0}-relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        if add_final_conv:
            main.add_module('final-{0}-{1}-conv'.format(cndf, 1),
                            nn.Conv1d(cndf, nz, 4, 1, 0, bias=False))

        self.main = main

    def forward(self, input):

        output = self.main(input)

        return output

##
class Decoder_1D(nn.Module):
    """
    DCGAN DECODER NETWORK
    """
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(Decoder_1D, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module('initial-{0}-{1}-convt'.format(nz, cngf),
                        nn.ConvTranspose1d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial-{0}-batchnorm'.format(cngf),
                        nn.BatchNorm1d(cngf))
        main.add_module('initial-{0}-relu'.format(cngf),
                        nn.ReLU(True))

        csize, _ = 4, cngf
        while csize < isize // 2:
            main.add_module('pyramid-{0}-{1}-convt'.format(cngf, cngf // 2),
                            nn.ConvTranspose1d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(cngf // 2),
                            nn.BatchNorm1d(cngf // 2))
            main.add_module('pyramid-{0}-relu'.format(cngf // 2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}-{1}-conv'.format(t, cngf),
                            nn.Conv1d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cngf),
                            nn.BatchNorm1d(cngf))
            main.add_module('extra-layers-{0}-{1}-relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final-{0}-{1}-convt'.format(cngf, nc),
                        nn.ConvTranspose1d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final-{0}-tanh'.format(nc),
                        nn.Tanh())
        self.main = main

    def forward(self, input):

        output = self.main(input)
        return output


class BiVi_NetG(nn.Module):
    
    def __init__(self, opt):
        super(BiVi_NetG, self).__init__()
        self.enc = Encoder_1D(isize= opt.z_size, 
                                nz=opt.feature_ch,
                                nc=opt.z_ch, 
                                ndf=opt.ndf,
                                ngpu=opt.ngpu, 
                                n_extra_layers=opt.extralayers)
        self.dec1 = Decoder(isize=opt.img_size,
                            nz=opt.feature_ch,
                            nc=opt.img_ch,
                            ngf=opt.ngf,
                            ngpu=opt.ngpu,
                            n_extra_layers=opt.extralayers)
        self.dec2 = Decoder_1D(isize= opt.signal_size, 
                                nz=opt.feature_ch,
                                nc=opt.signal_ch, 
                                ngf=opt.ngf,
                                ngpu=opt.ngpu, 
                                n_extra_layers=opt.extralayers)
    
    
    def forward(self, z, x):
        
        feature = self.enc(z)

        x_hat = self.dec2(feature)
        reconstruct_x = extract_and_reconstruct(x)
        
        x_hat = x_hat+reconstruct_x
        
        feature = feature.unsqueeze(-1)
        
        cwt_hat = self.dec1(feature)
        
        return x_hat, cwt_hat

class BiVi_NetE(nn.Module):
    
    def __init__(self, opt):
        super(BiVi_NetE, self).__init__()
        
        self.enc = Encoder_1D(isize=opt.signal_size,
                                nz=opt.z_size,
                                nc=opt.signal_ch,
                                ndf=opt.ndf,
                                ngpu=opt.ngpu,
                                n_extra_layers=opt.extralayers)
        
    def forward(self, x):
        reconstruct_x = extract_and_reconstruct(x)
        x = x-reconstruct_x
        
        z_hat = self.enc(x)
        z_hat = z_hat.permute(0,2,1)
        
        return z_hat
    
class BiVi_NetD(nn.Module):
    
    def __init__(self, opt):
        super(BiVi_NetD, self).__init__()
        
        
        self.feat_z = Encoder_1D(isize= opt.z_size, 
                                nz=int(opt.feature_ch/2),
                                nc=opt.z_ch, 
                                ndf=opt.ndf,
                                ngpu=opt.ngpu, 
                                n_extra_layers=opt.extralayers)
        self.feat_x = Encoder_1D(isize=opt.signal_size,
                                nz=int(opt.feature_ch/2),
                                nc=opt.signal_ch,
                                ndf=opt.ndf,
                                ngpu=opt.ngpu,
                                n_extra_layers=opt.extralayers)
        
        
        
        self.classifier = Encoder_1D(isize=opt.feature_ch,
                                nz=1,
                                nc=1,
                                ndf=opt.ndf,
                                ngpu=opt.ngpu,
                                n_extra_layers=opt.extralayers)
        self.classifier.add_module('Sigmoid', nn.Sigmoid())
        
    def forward(self, z, x):
        z_feature = self.feat_z(z)
        x_feature = self.feat_x(x)
        
        feature = torch.cat((z_feature, x_feature),dim=1)
        
        feature = feature.permute(0,2,1)
        
        result = self.classifier(feature)
        result = result.view(-1, 1).squeeze(1)
        
        return result

if __name__ =='__main__':
    
    opt = Options().parse()

    netg = NetG(opt=opt)
    nete = NetE(opt=opt)
    netd = NetD(opt=opt)
    
    z = torch.randn(opt.batchsize, opt.z_ch, opt.z_size)
    x = torch.randn(opt.batchsize, opt.signal_ch, opt.signal_size)

    z_hat = nete(x)

    x_hat, cwt_hat = netg(z, x)
    
    result = netd(z_hat, x_hat)
    
    print(f'result.size : {result.size()}')
    print(f'result : {result}')
    
    