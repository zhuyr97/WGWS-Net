import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
import matplotlib.pyplot as plt

def feature_save1(tensor, name):
    # tensor = torchvision.utils.make_grid(tensor.transpose(1,0))
    # tensor = torch.mean(tensor,dim=1).repeat(3,1,1)
    if not os.path.exists(str(name)):
        os.makedirs(str(name))
    for i in range(tensor.shape[1]):
        inp = tensor[:,i,:,:].detach().cpu().numpy().transpose(1,2,0)
        inp = np.clip(np.abs(inp),0,1)
        inp = (inp-np.min(inp))/(np.max(inp)-np.min(inp))
        inp = np.squeeze(inp)
        plt.figure()
        plt.imshow(inp)
        plt.savefig(str(name) + '/' + str(i) + '.png')

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias= True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        #layers = list()
        self.transpose= transpose
        if self.transpose:
            padding = kernel_size // 2 -1
            self.layer = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias)
        else:
            self.layer = nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias)

        self.relu= relu
        if self.relu:
            self.act = nn.GELU()

    def forward(self, x):
        if self.relu:
            # if self.transpose:
            #     return self.act(self.layer(x))
            # else:
            return self.act(self.layer(x))
        else:
            # if self.transpose:
            #     return self.layer(x)
            # else:
            return self.layer(x)


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = BasicConv(in_channel=Cin, out_channel=G, kernel_size=kSize, stride=1, relu=True)
    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDBlock(nn.Module):
    def __init__(self, in_channel, out_channel, nConvLayers=3):
        super(RDBlock, self).__init__()
        G0 = in_channel
        G = in_channel
        C = nConvLayers

        self.conv0 = RDB_Conv(G0 , G)
        self.conv1 = RDB_Conv(G0 + 1 * G , G)
        self.conv2 = RDB_Conv(G0 + 2 * G , G)
        # Local Feature Fusion
        self.LFF = BasicConv(in_channel=G0 + C * G, out_channel=out_channel, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        out = self.conv0(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.LFF(out) + x
        return out


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()
        self.layer1 = RDBlock(out_channel, out_channel)
        self.layer2 = RDBlock(out_channel, out_channel)
        self.layer3 = RDBlock(out_channel, out_channel)
        self.layer4 = RDBlock(out_channel, out_channel)
        self.layer5 = RDBlock(out_channel, out_channel)
        self.layer6 = RDBlock(out_channel, out_channel)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        return out


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()
        # layers = [RDBlock(channel, channel) for _ in range(num_res)]
        # self.layers = nn.Sequential(*layers)
        self.layer1 = RDBlock(channel, channel)
        self.layer2 = RDBlock(channel, channel)
        self.layer3 = RDBlock(channel, channel)
        self.layer4 = RDBlock(channel, channel)
        self.layer5 = RDBlock(channel, channel)
        self.layer6 = RDBlock(channel, channel)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        return out


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()

        self.conv1 = BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True)
        self.conv2 = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        out = self.conv1(x)
        out = self.conv2(out)

        return out


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.layer1 = BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True)
        self.layer2 = BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True)
        self.layer3 = BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True)
        self.layer4 = BasicConv(out_plane // 2, out_plane-3, kernel_size=1, stride=1, relu=True)
        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        x = torch.cat([x,out], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out



class UNet(nn.Module):
    def __init__(self, base_channel=24, num_res=6):
        super(UNet, self).__init__()

        base_channel = base_channel

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])
        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1),
            AFF(base_channel * 7, base_channel*2),
            AFF(base_channel * 7, base_channel * 4)
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)
    def forward(self, x):

        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)
        #outputs.append(res1.flatten(1))

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)

        z = self.Encoder[2](z)

        #-----------------------------inter-------------------#
        z12 = F.interpolate(res1, scale_factor=0.5)
        z14 = F.interpolate(res1, scale_factor=0.25)
        z21 = F.interpolate(res2, scale_factor=2)
        z24 = F.interpolate(res2, scale_factor=0.5)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)
        z = self.AFFs[2](z14, z24, z)
        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)
        # -----------------------------inter-------------------#

        z = self.Decoder[0](z)
        z = self.feat_extract[3](z)
        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z = self.feat_extract[4](z)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        return z + x



if __name__ == "__main__":
    model = UNet(base_channel=21, num_res=6)
    count = 0
    for name,module in model.named_modules():
        print(count,'-------------',name)
        count +=1
    from functools import partial

    input = torch.randn(1, 3, 64, 64)
    output = model(input)
    print('-'*50)
    print(output.shape)
    print('#generator parameters:', sum(param.numel() for param in model.parameters()))
