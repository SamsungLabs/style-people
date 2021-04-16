import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Function
from torch.nn.utils import spectral_norm
from torch.optim import Adam

from utils.common import to_sigm, to_tanh
from utils.bbox import compute_bboxes_from_keypoints, crop_resize_image, crop_and_resize

from models.styleganv2.modules import (
    PixelNorm, make_kernel, Upsample, Downsample, Blur, EqualConv2d,
    EqualLinear, ScaledLeakyReLU, ModulatedConv2d, NoiseInjection,
    ConstantInput, StyledConv, ToRGB, ConvLayer, ResBlock
)


class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--face_size', type=int, default=128)
        parser.add('--fd_input_channels', type=int, default=4)
        parser.add('--lr_dis_face', type=float, default=0.002)

    @staticmethod
    def get_net(args):
        net = Discriminator(args.face_size, input_channels=args.fd_input_channels, alternating=args.alternating, device=args.device).to(args.device)
        return net

    @staticmethod
    def get_optimizer(discriminator, args):
        d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
        optimizer = optim.Adam(
            discriminator.parameters(),
            lr=args.lr_dis_face * d_reg_ratio,
            betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
        )

        return dict(dface_optim=optimizer)


class Discriminator(nn.Module):
    def __init__(self, size, input_channels=3, channel_multiplier=4, alternating=False, blur_kernel=[1, 3, 3, 1], device='cuda'):
        super().__init__()

        self.input_channels = input_channels
        self.face_size = size
        self.device = device
        self.alternating = alternating

        self.fake_stash = None
        self.real_stash = None

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

        convs = [ConvLayer(input_channels, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
            EqualLinear(channels[4], 1),
        )

    def make_valid_mask(self, face_kp):
        valid_mask = (face_kp[..., 0] < 0).sum(dim=1) <= 0
        return valid_mask

    def get_stashed(self, rgb, C, step):
        if step == 'fake':
            if self.fake_stash is None:
                face = torch.zeros_like(rgb[:, :, :self.face_size, :self.face_size])
                inp = torch.zeros(1, C, self.face_size, self.face_size).to(rgb.device)
                face.grad = None
                inp.grad = None
                return face, inp
            else:
                self.fake_stash[0].grad = None
                self.fake_stash[1].grad = None
                return self.fake_stash
        elif step == 'real':
            if self.real_stash is None:
                face = torch.zeros_like(rgb[:, :, :self.face_size, :self.face_size])
                inp = torch.zeros(1, C, self.face_size, self.face_size).to(rgb.device)
                face.grad = None
                inp.grad = None
                return face, inp
            else:
                return self.real_stash

    def stash(self, face, inp, step):
        if step == 'fake':
            self.fake_stash = (face.detach(), inp.detach())
        elif step == 'real':
            self.real_stash = (face.detach(), inp.detach())

    def pass_input(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = batch
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

    def extract_face(self, data_dict):
        data_dict_ = data_dict['main']
        rgb = data_dict_['rgb']
        segm = data_dict_['segm']
        landmarks = data_dict_['landmarks']
        

        # TODO
        if 'step' in data_dict:
            step = data_dict['step']
        else:
            step = 'real'

        face_kp = landmarks[:, 25:93]

        rgb = to_sigm(rgb) * segm
        rgb = to_tanh(rgb)

        valid_mask = self.make_valid_mask(face_kp)
        segm_cropped = None

        if valid_mask.sum() == 0:
            C = rgb.shape[1] + segm.shape[1]
            face, inp = self.get_stashed(rgb, C, step)
            inp.requires_grad = True

            score = self.pass_input(inp.contiguous())

        else:
            rgb = rgb[valid_mask]
            segm = segm[valid_mask]

            face_kp = face_kp[valid_mask]

            bboxes_estimate = compute_bboxes_from_keypoints(face_kp)

            face_size = (self.face_size, self.face_size)

            face = crop_and_resize(rgb, bboxes_estimate, face_size)
            segm_cropped = crop_and_resize(segm, bboxes_estimate, face_size)

            inp = torch.cat([face, segm_cropped], dim=1).detach()
            self.stash(face, inp, step)

        return inp, face, segm_cropped

    def forward(self, data_dict):
        return_dict = {}

        inp, face, segm_cropped = self.extract_face(data_dict)
        model_output = self.pass_input(inp.contiguous())

        if 'return_inp' in data_dict and data_dict['return_inp']:
            return_dict['dface_inp'] = inp
            inp.requires_grad = True
        
        # return results
        return_dict.update(dict(dface_score=model_output['score'], face=face, score=model_output['score']))
        return return_dict

    def extract_endpoints(self, data_dict):
        return_dict = {}

        inp, face, segm_cropped = self.extract_face(data_dict)
        model_output = self._extract_endpoints(inp.contiguous())

        if 'return_inp' in data_dict and data_dict['return_inp']:
            return_dict['dface_inp'] = inp
            inp.requires_grad = True
        
        # return results
        return_dict.update(dict(dface_score=model_output['score'], face=face, score=model_output['score'], endpoints=model_output['endpoints']))
        return return_dict


    def _extract_endpoints(self, input):
        endpoints = dict()

        out = input
        for conv in self.convs:
            out = conv(out)
            endpoints[f'reduction_{len(endpoints) + 1}'] = out

        batch, channel, height, width = out.shape
        group = batch
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)
        endpoints[f'reduction_{len(endpoints) + 1}'] = out

        out = out.view(batch, -1)
        out = self.final_linear(out)
        
        # return results
        out_dict = {
            'score': out,
            'endpoints': endpoints
        }

        return out_dict
