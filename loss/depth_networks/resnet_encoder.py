from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()



def convt_bn_relu(in_channels, out_channels, kernel_size, \
        stride=1, padding=0, output_padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(
        nn.ConvTranspose2d(in_channels,
                           out_channels,
                           kernel_size,
                           stride,
                           padding,
                           output_padding,
                           bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers = nn.Sequential(*layers)

    # initialize the weights
    for m in layers.modules():
        init_weights(m)

    return layers


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        ######################################
        # night public first conv
        ######################################
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.conv_shared = nn.Conv2d(512, 64, kernel_size=1)

        ##########################################
        # private encoder, day
        ##########################################
        self.encoder_day = resnets[num_layers](pretrained)
        self.conv_diff_day = nn.Conv2d(512, 64, kernel_size=1) #no bn after conv, so bias=true

        ##########################################
        # private encoder, night
        ##########################################
        self.encoder_night = resnets[num_layers](pretrained)
        self.conv_diff_night = nn.Conv2d(512, 64, kernel_size=1)

        ######################################
        # shared decoder (small decoder), use a simple de-conv to upsample the features with no skip connection
        ######################################
        self.convt5 = convt_bn_relu(in_channels=512,out_channels=256,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.convt4 = convt_bn_relu(in_channels=256,out_channels=128,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.convt3 = convt_bn_relu(in_channels=128,out_channels=64,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.convt2 = convt_bn_relu(in_channels=64,out_channels=64,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.convt1 = convt_bn_relu(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1,output_padding=1)
        self.convtf = nn.Conv2d(64,3,kernel_size=1,stride=1,padding=0)


    def forward(self, input_image, is_night, istrain):

        if istrain=='train':
            result = []
            input_data = (input_image - 0.45) / 0.225
            if is_night == 'day':
                # source private encoder, day
                private_feature = self.encoder_day.conv1(input_data)
                private_feature = self.encoder_day.bn1(private_feature)
                private_feature = self.encoder_day.relu(private_feature)
                private_feature = self.encoder_day.maxpool(private_feature)
                private_feature = self.encoder_day.layer1(private_feature)
                private_feature = self.encoder_day.layer2(private_feature)
                private_feature = self.encoder_day.layer3(private_feature)
                private_feature = self.encoder_day.layer4(private_feature)
                private_code = self.conv_diff_day(private_feature)
                private_gram = gram_matrix(private_feature)
                result.append(private_code)
                result.append(private_gram)

            elif is_night == 'night':
                # target private encoder, night
                private_feature = self.encoder_night.conv1(input_data)
                private_feature = self.encoder_night.bn1(private_feature)
                private_feature = self.encoder_night.relu(private_feature)
                private_feature = self.encoder_night.maxpool(private_feature)
                private_feature = self.encoder_night.layer1(private_feature)
                private_feature = self.encoder_night.layer2(private_feature)
                private_feature = self.encoder_night.layer3(private_feature)
                private_feature = self.encoder_night.layer4(private_feature)
                private_code = self.conv_diff_night(private_feature)

                private_gram = gram_matrix(private_feature)
                result.append(private_code)
                result.append(private_gram)


        # shared encoder
        self.features = []
        x = (input_image - 0.45) / 0.225
        if is_night=='day':
            x = self.encoder.conv1(x)
            x = self.encoder.bn1(x)
            self.features.append(self.encoder.relu(x))
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            self.features.append(self.relu(x))

        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        if istrain=='train':
            shared_code = self.conv_shared(self.features[-1])
            shared_gram = gram_matrix(self.features[-1])
            result.append(shared_code)  # use this to calculate loss of ortho
            result.append(shared_gram)
            result.append(self.features[-1])  # use this to calculate loss of similarity

            union_code = private_feature + self.features[-1]
            rec_code = self.convt5(union_code)
            rec_code = self.convt4(rec_code)
            rec_code = self.convt3(rec_code)
            rec_code = self.convt2(rec_code)
            rec_code = self.convt1(rec_code)
            rec_code = self.convtf(rec_code)
            result.append(rec_code)

            return self.features, result
        else:
            return self.features