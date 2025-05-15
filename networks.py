import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import *
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as spectral_norm_fn
from torch.nn.utils import weight_norm as weight_norm_fn
from position_encoding import build_position_encoding
from transformer import Transformer
from torchvision import models
from iconv import IConv
import numpy as np


class G_Net(nn.Module):
    def     __init__(self, in_ch=3,out_ch=3, residual_blocks=4):
        super(G_Net, self).__init__()

        self.pos_embeding = nn.Parameter(torch.randn(4096, 1, 1024))
        num_heads = 4
        enc_layers = 1
        dec_layers = 1

        hidden_dim = 1024
        dim_feedforward = 1024


        self.transformer = Transformer(
            d_model=hidden_dim,
            dropout=0.1,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=False,
            return_intermediate_dec=False,
        )

        # ---------------------------------------------------------------

        cnums1 = 64
        self.stage3 = RSU5(in_ch+1, cnums1, cnums1 * 2)
        self.stage4 = RSU4(cnums1 * 2, cnums1, cnums1 * 4)
        self.stage5 = RSU4F(cnums1 * 4, cnums1 * 2, cnums1 * 8)
        self.stage6 = RSU4F(cnums1 * 8, cnums1 * 4, cnums1 * 8)

        self.stage5d = RSU4F(cnums1 * 8, cnums1 * 2, cnums1 * 4)
        self.stage4d = RSU4(cnums1 * 4, cnums1, cnums1 * 2)
        self.stage3d = RSU5(cnums1 * 2, cnums1//2, cnums1)

        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)

        self.mask_side4 = REBNCONV(cnums1 * 8, cnums1 * 8, dirate=1, activation='sigmoid')
        self.mask_side5 = REBNCONV(cnums1 * 4, cnums1 * 4, dirate=1, activation='sigmoid')
        self.mask_side6 = REBNCONV(cnums1 * 2, cnums1 * 2, dirate=1, activation='sigmoid')


    def forward(self,x, mask): #x应该是输入图片
        hx = (x * (1 - mask).float()) + mask
        hx = torch.cat((hx, mask), 1)


        # ---------------------------------------------------------------------

        # stage 3
        hx3 = self.stage3(hx)
        # print(hx3.shape)
        # hx = self.pool34(hx3)
        hx = F.interpolate(hx3, scale_factor=0.5, mode='bilinear')


        # stage 4
        hx4 = self.stage4(hx)
        # print(hx4.shape)
        # hx = self.pool45(hx4)
        hx = F.interpolate(hx4, scale_factor=0.5, mode='bilinear')

        bs, c, h, w = hx4.shape
        # src = src.reshape(bs, c, h*w)

        hx4 = hx4.reshape(bs, c, h // 64, 64, w // 64, 64)
        hx4 = hx4.permute(0, 1, 2, 4, 3, 5)
        # # b,h,w,c,8,8
        hx4 = hx4.reshape(bs, c * h // 64 * w // 64, 64 * 64)
        # src += pos
        hx4 = hx4.permute(2, 0, 1)

        # hx4 = self.linear_en1(hx4)

        hx4 = hx4 + self.pos_embeding

        hx4 = self.transformer(hx4)

        # hx4 = self.linear_de1(hx4)

        hx4 = hx4.permute(1, 2, 0)
        hx4 = hx4.reshape(bs, c, h // 64, w // 64, 64, 64)
        hx4 = hx4.permute(0, 1, 2, 4, 3, 5)
        hx4 = hx4.reshape(bs, c, h, w)


        # stage 5
        hx5 = self.stage5(hx)
        # print(hx5.shape)
        # hx = self.pool56(hx5)
        hx = F.interpolate(hx5, scale_factor=0.5, mode='bilinear')


        # stage 6
        hx6 = self.stage6(hx)
        # print(hx6.shape)
        hx6up = _upsample_like(hx6, hx5)
        hx_mask4 = self.mask_side4(hx6up)
        hx6up = hx6up * hx_mask4 + hx5 * (1 - hx_mask4)

        # -------------------- decoder --------------------
        hx5d = self.stage5d(hx6up)
        # print(hx5d.shape)
        hx5dup = _upsample_like(hx5d, hx4)
        hx_mask5 = self.mask_side5(hx5dup)
        hx5dup = hx5dup * hx_mask5 + hx4 * (1 - hx_mask5)

        hx4d = self.stage4d(hx5dup)
        # print(hx4d.shape)
        hx4dup = _upsample_like(hx4d, hx3)
        hx_mask6 = self.mask_side6(hx4dup)
        hx4dup = hx4dup * hx_mask6 + hx3 * (1 - hx_mask6)

        hx3d = self.stage3d(hx4dup)
        # print(hx3d.shape)

        d3 = self.side3(hx3d)
        d3 = (torch.tanh(d3) + 1) / 2

        return d3


# original D
class D_Net(nn.Module):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True):
        super(D_Net, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )


    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]

# # original D
# class D_Net(nn.Module):
#     def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True):
#         super(D_Net, self).__init__()
#         self.use_sigmoid = use_sigmoid
#
#         self.conv1 = nn.Sequential(
#             spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=5, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#
#         self.conv2 = nn.Sequential(
#             spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#
#         self.conv3 = nn.Sequential(
#             spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#
#         self.conv4 = nn.Sequential(
#             spectral_norm(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#
#         self.conv5 = nn.Sequential(
#             spectral_norm(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#
#         self.conv6 = nn.Sequential(
#             spectral_norm(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#
#
#     def forward(self, x):
#         conv1 = self.conv1(x)
#         conv2 = self.conv2(conv1)
#         conv3 = self.conv3(conv2)
#         conv4 = self.conv4(conv3)
#         conv5 = self.conv5(conv4)
#         conv6 = self.conv6(conv5)
#
#         outputs = conv6
#         if self.use_sigmoid:
#             outputs = torch.sigmoid(conv6)
#
#         outputs = outputs.view(outputs.size(0), -1)
#
#         return outputs, [conv1, conv2, conv3, conv4, conv5, conv6]


class D_Net1(nn.Module):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True):
        super(D_Net1, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=not use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=not use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv6 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1,
                                    bias=not use_spectral_norm), 
        )


    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)

        outputs = conv6
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv6)

        return outputs, [conv1, conv2, conv3, conv4, conv5, conv6]


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=True):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=256, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ELU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, weight_norm='none', norm='none',
                 activation='elu', pad_type='replicate'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'none':
            self.pad = None
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = out_channels
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        if weight_norm == 'sn':
            self.weight_norm = spectral_norm_fn
        elif weight_norm == 'wn':
            self.weight_norm = weight_norm_fn
        elif weight_norm == 'none':
            self.weight_norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(weight_norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding=0, dilation=dilation,
                              bias=self.use_bias)

        if self.weight_norm:
            self.conv = self.weight_norm(self.conv)

    def forward(self, x):
        if self.pad:
            x = self.conv(self.pad(x))
        else:
            x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x



def _upsample_like(src,tar): #对src进行上采样，使其尺寸与tar张量的空间维度相匹配

    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src


### RSU-5 ###
class RSU5(nn.Module):#UNet05DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        # self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        # self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        # self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        # hx = self.pool1(hx1)
        hx = F.interpolate(hx1, scale_factor=0.5, mode='bilinear')

        hx2 = self.rebnconv2(hx)
        # hx = self.pool2(hx2)
        hx = F.interpolate(hx2, scale_factor=0.5, mode='bilinear')

        hx3 = self.rebnconv3(hx)
        # hx = self.pool3(hx3)
        hx = F.interpolate(hx3, scale_factor=0.5, mode='bilinear')

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin




### RSU-4 ###
class RSU4(nn.Module):#UNet04DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        # self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        # self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = F.interpolate(hx1, scale_factor=0.5, mode='bilinear')
        # hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = F.interpolate(hx2, scale_factor=0.5, mode='bilinear')
        # hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin


### RSU-4F ###
class RSU4F(nn.Module):#UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx2d = self.rebnconv2d(torch.cat((hx3d,hx2),1))
        hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))

        return hx1d + hxin

class REBNCONV(nn.Module):  # RE BN CONV ;dirate=dilation rate；指空洞卷积的核中元素距离；1代表着紧挨着
    def __init__(self,in_ch=3,out_ch=3,dirate=1, norm='none', activation='elu'):
        super(REBNCONV,self).__init__()

        norm_dim = out_ch
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)


        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)


        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        # self.bn_s1 = nn.BatchNorm2d(out_ch)
        # self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):

        hx = x
        hx = self.conv_s1(hx)
        if self.norm:
            hx = self.norm(hx)
        if self.activation:
            hx = self.activation(hx)
        # xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return hx


if __name__ == '__main__':
    print("No Abnormal!")
