import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Function

from models.styleganv2.modules import (
    PixelNorm, make_kernel, Upsample, Downsample, Blur, EqualConv2d,
    EqualLinear, ScaledLeakyReLU, ModulatedConv2d, NoiseInjection,
    ConstantInput, StyledConv, ToRGB, ConvLayer, ResBlock
)


class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--image_size', type=int, default=128)
        parser.add('--input_channels', type=int, default=3)
        parser.add('--lr_dis', type=float, default=0.002)
        parser.add('--fadein_steps', type=float, default=5000)
        parser.add('--fadestep_every', type=float, default=10)

    @staticmethod
    def get_net(args):
        net = Discriminator(args.image_size, input_channels=args.input_channels, 
            fadein_steps=args.fadein_steps, fadestep_every=args.fadestep_every).to(args.device)
        return net

    @staticmethod
    def get_optimizer(discriminator, args):
        d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
        d_optim = optim.Adam(
            discriminator.parameters(),
            lr=args.lr_dis * d_reg_ratio,
            betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
        )
        return dict(d_optim=d_optim)



class Discriminator(nn.Module):
    def __init__(self, size, input_channels=3, channel_multiplier=4, blur_kernel=[1, 3, 3, 1], fadein_steps=5000, fadestep_every=10):
        super().__init__()

        self.fadein_steps = fadein_steps
        self.fadestep_every = fadestep_every

        self.input_channels = input_channels

        channels = {
            4: min(128 * channel_multiplier, 512),
            8: min(128 * channel_multiplier, 512),
            16: min(128 * channel_multiplier, 512),
            32: min(128 * channel_multiplier, 512),
            64: 128 * channel_multiplier,
            128: 64 * channel_multiplier,
            256: 32 * channel_multiplier,
            512: 16 * channel_multiplier,
            1024: 8 * channel_multiplier,
        }
        size_lowres = size // 2

        highres_tail = [ConvLayer(input_channels, channels[size], 1)]
        highres_tail.append(ResBlock(channels[size],channels[size_lowres], blur_kernel))
        self.highres_tail = nn.Sequential(*highres_tail)

        lowres_tail = [ConvLayer(input_channels, channels[size_lowres], 1)]
        self.lowres_tail = nn.Sequential(*lowres_tail)

        log_size = int(math.log(size_lowres, 2))

        in_channel = channels[size_lowres]

        convs = []
        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 2
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
            EqualLinear(channels[4], 1),
        )

        self.size = size
        self.size_lowres = size_lowres

        self.i = 0

    def load_prev_scale(self, state_dict):
        state_dict_mod = dict()
        for k, v in state_dict.items():
            if k.startswith('convs.0'):
                k = k.replace('convs.0', 'lowres_tail.0')
            elif k.startswith('convs'):
                parts = k.split('.')
                cind = int(parts[1]) - 1
                parts[1] = str(cind)
                k = '.'.join(parts)
            state_dict_mod[k] = v
        conflicts = self.load_state_dict(state_dict_mod, strict=False)
        print(conflicts)


    def forward(self, data_dict):
        input = data_dict['input']
        iter_id = data_dict['iter']
        out = self.highres_tail(input)

        if iter_id < self.fadein_steps:
            alfa = 1./ (self.fadein_steps//self.fadestep_every) * (iter_id//self.fadestep_every)
            lr_input = torch.nn.functional.interpolate(input, size=(self.size_lowres, self.size_lowres), mode='bilinear')
            lr_out = self.lowres_tail(lr_input)
            hr_out = out
            out = hr_out*alfa + lr_out*(1.-alfa)

            # out_dict = dict(input=input, lr_input=lr_input, hr_out=hr_out, lr_out=lr_out, out=out)
            # if self.input_channels == 4:
            #     torch.save(out_dict, f'/Vol1/dbstore/datasets/a.grigorev/temp/sgv2_ups/{self.i:04d}.pth')
            # else:
            #     torch.save(out_dict, f'/Vol1/dbstore/datasets/a.grigorev/temp/sgv2_ups_bin/{self.i:04d}.pth')
            # self.i += 1

        out = self.convs(out)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)
        out = out.view(batch, -1)
        out = self.final_linear(out)
        
        # return results
        out_dict = {
            'score': out
        }

        return out_dict