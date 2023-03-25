
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


#########################################################################################################################################

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return (-1) * ssim_map.mean()
    else:
        return (-1) * ssim_map.mean(1).mean(1).mean(1)


class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return 1 + _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


###########################################################################################################################

class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg = models.vgg19(pretrained=False)
        vgg.load_state_dict(torch.load('//gdata2/zhuyr/VGG/vgg19-dcbb9e9d.pth'))
        vgg.eval()
        vgg_pretrained_features = vgg.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(3):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(3, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().to(device)
        self.criterion = nn.MSELoss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)

    def forward(self, x, y):
        while x.size()[3] > 4096:
            x, y = self.downsample(x), self.downsample(y)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        # loss = self.criterion(x_vgg, y_vgg.detach())
        for i in range(len(x_vgg)):
            loss += self.weights[i]*self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


############################################################################################################################3


class GradientLoss(nn.Module):
    """Gradient Histogram Loss"""
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.bin_num = 64
        self.delta = 0.2
        self.clip_radius = 0.2
        assert(self.clip_radius>0 and self.clip_radius<=1)
        self.bin_width = 2*self.clip_radius/self.bin_num
        if self.bin_width*255<1:
            raise RuntimeError("bin width is too small")
        self.bin_mean = np.arange(-self.clip_radius+self.bin_width*0.5, self.clip_radius, self.bin_width)
        self.gradient_hist_loss_function = 'L2'
        # default is KL loss
        if self.gradient_hist_loss_function == 'L2':
            self.criterion = nn.MSELoss()
        elif self.gradient_hist_loss_function == 'L1':
            self.criterion = nn.L1Loss()
        else:
            self.criterion = nn.KLDivLoss()

    def get_response(self, gradient, mean):
        # tmp = torch.mul(torch.pow(torch.add(gradient, -mean), 2), self.delta_square_inverse)
        s = (-1) / (self.delta ** 2)
        tmp = ((gradient - mean) ** 2) * s
        return torch.mean(torch.exp(tmp))

    def get_gradient(self, src):
        right_src = src[:, :, 1:, 0:-1]     # shift src image right by one pixel
        down_src = src[:, :, 0:-1, 1:]      # shift src image down by one pixel
        clip_src = src[:, :, 0:-1, 0:-1]    # make src same size as shift version
        d_x = right_src - clip_src
        d_y = down_src - clip_src

        return d_x, d_y

    def get_gradient_hist(self, gradient_x, gradient_y):
        lx = None
        ly = None
        for ind_bin in range(self.bin_num):
            fx = self.get_response(gradient_x, self.bin_mean[ind_bin])
            fy = self.get_response(gradient_y, self.bin_mean[ind_bin])
            fx = torch.cuda.FloatTensor([fx])
            fy = torch.cuda.FloatTensor([fy])

            if lx is None:
                lx = fx
                ly = fy
            else:
                lx = torch.cat((lx, fx), 0)
                ly = torch.cat((ly, fy), 0)
        # lx = torch.div(lx, torch.sum(lx))
        # ly = torch.div(ly, torch.sum(ly))
        return lx, ly

    def forward(self, output, target):
        output_gradient_x, output_gradient_y = self.get_gradient(output)
        target_gradient_x, target_gradient_y = self.get_gradient(target)

        output_gradient_x_hist, output_gradient_y_hist = self.get_gradient_hist(output_gradient_x, output_gradient_y)
        target_gradient_x_hist, target_gradient_y_hist = self.get_gradient_hist(target_gradient_x, target_gradient_y)
        # loss = self.criterion(output_gradient_x_hist, target_gradient_x_hist) + self.criterion(output_gradient_y_hist, target_gradient_y_hist)
        loss = self.criterion(output_gradient_x,target_gradient_x)+self.criterion(output_gradient_y,target_gradient_y)
        return loss

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-4):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        #diff = x.to('cuda:0') - y.to('cuda:0')
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class CharbonnierLoss1(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-4):
        super(CharbonnierLoss1, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        #diff = x.to('cuda:0') - y.to('cuda:0')
        diff = x - y
        loss = torch.mean((diff * diff) + (self.eps*self.eps))
        return loss

import loss.depth_networks as networks

class depth_loss(nn.Module):
    def __init__(self):
        super(depth_loss, self).__init__()
        self.criterion = nn.L1Loss()

        self.encoder = networks.ResnetEncoder(18, False)
        encoder_path = '/ghome/zhuyr/UDC_codes/other_methods/ADDS-DepthNet-main/pretrained_model/encoder.pth'
        loaded_dict_enc = torch.load(encoder_path, map_location=device)

        # extract the height and width of image that this model was trained with
        self.feed_height = loaded_dict_enc['height']
        self.feed_width = loaded_dict_enc['width']
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)
        self.encoder.to(device)
        self.encoder.eval()

        #print("   Loading pretrained decoder")
        self.depth_decoder = networks.DepthDecoder(num_ch_enc=self.encoder.num_ch_enc,
                                                   scales=range(4))
        depth_decoder_path = '/ghome/zhuyr/UDC_codes/other_methods/ADDS-DepthNet-main/pretrained_model/depth.pth'
        loaded_dict = torch.load(depth_decoder_path, map_location=device)
        self.depth_decoder.load_state_dict(loaded_dict)
        self.depth_decoder.to(device)
        self.depth_decoder.eval()

    def forward(self, x, y):
        # print('self.feed_height,self.feed_width',self.feed_height,self.feed_width) #256 512
        x_in = F.interpolate(x,size=(self.feed_height,self.feed_width),mode='bilinear')
        # torch.nn.functional.interpolate(input, size=(self.feed_height,self.feed_width), scale_factor=None, mode='bilinear', align_corners=None,
        #                             recompute_scale_factor=None)
        y_in = F.interpolate(y,size=(self.feed_height,self.feed_width),mode='bilinear')
        # torch.nn.functional.interpolate(input, size=(self.feed_height,self.feed_width), scale_factor=None, mode='bilinear', align_corners=None,
        #                             recompute_scale_factor=None)

        x_out = self.depth_decoder(self.encoder(x_in, 'day', 'val'))
        y_out = self.depth_decoder(self.encoder(y_in, 'day', 'val'))
        #print('len x_out:',len(x_out))
        x_out = x_out[("disp", 0)]
        y_out = y_out[("disp", 0)]

        return self.criterion(x_out,y_out)



class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        self.kernel = self.kernel #.to(device)
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)
        down        = filtered[:,:,::2,::2]
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4
        filtered    = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y):
        #loss = self.loss(self.laplacian_kernel(x.to('cuda:0')), self.laplacian_kernel(y.to('cuda:0')))
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))

        return loss

class fftLoss(nn.Module):
    def __init__(self):
        super(fftLoss, self).__init__()

    def forward(self, x, y):
        #diff = torch.fft.fft2(x.to('cuda:0')) - torch.fft.fft2(y.to('cuda:0'))
        diff = torch.fft.fft2(x) - torch.fft.fft2(y)
        loss = torch.mean(abs(diff))
        return loss


class fftLoss_old_version(nn.Module):
    def __init__(self):
        super(fftLoss_old_version, self).__init__()

    def forward(self, x, y):
        #diff = torch.fft.fft2(x.to('cuda:0')) - torch.fft.fft2(y.to('cuda:0'))
        diff = torch.rfft(x, signal_ndim=2, normalized=False, onesided=False) - torch.rfft(y, signal_ndim=2, normalized=False, onesided=False)
        #torch.fft.fft2(x) - torch.fft.fft2(y)
        loss = torch.mean(abs(diff))
        return loss


#   train_out_fft = torch.rfft(train_output[2], signal_ndim=2, normalized=False, onesided=False)
#   train_labels_fft = torch.rfft(labels, signal_ndim=2, normalized=False, onesided=False)

#maoyy_stark_1.0