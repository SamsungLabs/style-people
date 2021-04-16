import os

import torch
from torch import nn

import utils.common
from models.styleganv2.modules import EqualConv2d
from models.styleganv2.op import FusedLeakyReLU
from models.common.unet import Unetv2


class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--image_size', type=int, default=256)
        parser.add('--style_dim', type=int, default=512)
        parser.add('--n_mlp', type=int, default=8)
        parser.add('--output_channels', type=int, default=3)

    @staticmethod
    def get_net(args):
        net = Renderer()
        net = net.to(args.device)
        return net
    
    
class Renderer(nn.Module):
    def __init__(self, in_channels=18, segm_channels=3, ngf=64, normalization='batch'):
        super().__init__()

        n_out = 16
        self.model = Unetv2(in_channels=in_channels, classes=n_out, ngf=ngf, same=False)
        norm_layer = nn.InstanceNorm2d if normalization == 'instance' else nn.BatchNorm2d

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        upsample_layers = []
        upsample_layers.append(EqualConv2d(n_out, n_out, 3, 1, 1, bias=False))
        upsample_layers.append(norm_layer(n_out, affine=True))
        upsample_layers.append(FusedLeakyReLU(n_out))
        upsample_layers.append(EqualConv2d(n_out, n_out, 3, 1, 1, bias=False))
        upsample_layers.append(norm_layer(n_out, affine=True))
        upsample_layers.append(FusedLeakyReLU(n_out))
        upsample_layers.append(EqualConv2d(n_out, n_out, 3, 1, 1, bias=False))
        upsample_layers.append(norm_layer(n_out, affine=True))
        upsample_layers.append(FusedLeakyReLU(n_out))
        upsample_layers.append(EqualConv2d(n_out, n_out, 3, 1, 1, bias=False))
        upsample_layers.append(norm_layer(n_out, affine=True))
        upsample_layers.append(FusedLeakyReLU(n_out))
        self.upsample_convs = nn.Sequential(*upsample_layers)

        self.rgb_head = nn.Sequential(
            norm_layer(n_out, affine=True),
            FusedLeakyReLU(n_out),
            EqualConv2d(n_out, n_out, 3, 1, 1, bias=False),
            norm_layer(n_out, affine=True),
            FusedLeakyReLU(n_out),
            EqualConv2d(n_out, n_out, 3, 1, 1, bias=False),
            norm_layer(n_out, affine=True),
            FusedLeakyReLU(n_out),
            EqualConv2d(n_out, 3, 3, 1, 1, bias=True),
            nn.Tanh())

        self.segm_head = nn.Sequential(
            norm_layer(n_out, affine=True),
            FusedLeakyReLU(n_out),
            EqualConv2d(n_out, 3, 3, 1, 1, bias=True),
            nn.Sigmoid())

        self.segm_channels = segm_channels

        self.last_iter = None

    def forward(self, data_dict):
        assert 'uv' in data_dict
        assert 'nrender' in data_dict

        uv = data_dict['uv']
        nrender = data_dict['nrender']
        uvmask = ((uv > -10).sum(dim=1, keepdim=True) > 0).float()

        inp = torch.cat([nrender, uv], dim=1)
        out_lr = self.model(inp)
        out_lr = self.upsample(out_lr)
        out = self.upsample_convs(out_lr)

        rgb = self.rgb_head(out)
        segm = self.segm_head(out)

        segm = segm[:, :self.segm_channels]
        segm_fg = segm[:, :1]

        segm_H = segm_fg.shape[2]
        mask_H = uvmask.shape[2]
        if segm_H != mask_H:
            uvmask = torch.nn.functional.interpolate(uvmask, size=(segm_H, segm_H))

        segm_fg = (segm_fg + uvmask).clamp(0., 1.)

        if 'crop_mask' in data_dict:
            segm_fg = segm_fg * data_dict['crop_mask']

        if 'background' in data_dict:
            background = data_dict['background']
            rgb_segm = utils.common.to_sigm(rgb) * segm_fg + background * (1. - segm_fg)
        else:
            rgb_segm = utils.common.to_sigm(rgb) * segm_fg
        rgb_segm = utils.common.to_tanh(rgb_segm)

        out_dict = dict(fake_rgb=rgb_segm, fake_segm=segm_fg, r_input=inp)

        return out_dict