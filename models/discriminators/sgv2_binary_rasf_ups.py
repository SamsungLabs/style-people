import math

import torch
from torch import nn, optim
from discriminators import styleganv2_ups
import random

class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--image_size', type=int, default=256)
        parser.add('--target_size', type=int)
        parser.add('--bidis_input_channels', type=int, default=8)
        parser.add('--bidis_channel_multiplier', type=int, default=6)
        parser.add('--lr_dis_pair', type=float, default=0.002)
        parser.add('--rasf_ratio', type=float, default=0.2)
        parser.add('--fadein_steps', type=float, default=5000)
        parser.add('--fadestep_every', type=float, default=10)

    @staticmethod
    def get_net(args):
        if not hasattr(args, 'target_size') or args.target_size is None:
            args.target_size = args.image_size
            
        net = Discriminator(args.target_size, input_channels=args.bidis_input_channels, 
            channel_multiplier=args.bidis_channel_multiplier, rasf_ratio=args.rasf_ratio, 
            fadein_steps=args.fadein_steps, fadestep_every=args.fadestep_every).to(args.device)
        return net

    @staticmethod
    def get_optimizer(discriminator, args):
        d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
        optimizer = optim.Adam(
            discriminator.parameters(),
            lr=args.lr_dis_pair * d_reg_ratio,
            betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
        )

        return dict(dpair_optim=optimizer)


class Discriminator(nn.Module):
    def __init__(self, size, input_channels=3, channel_multiplier=4, rasf_ratio=0.2,
                                        fadein_steps=5000, fadestep_every=10):
        super().__init__()

        self.model = styleganv2_ups.Discriminator(size, input_channels, channel_multiplier, 
            fadein_steps=fadein_steps, fadestep_every=fadestep_every)

        self.rasf_ratio = rasf_ratio

    def load_prev_scale(self, state_dict):
        state_dict = {k[6:]:v for k,v in state_dict.items()}
        self.model.load_prev_scale(state_dict)
        print('BINARY RASF LOADED')

    def forward(self, data_dict):
        if 'pair' not in data_dict:
            return dict()

        rasf_val = random.random()
        if 'rasf' in data_dict and rasf_val < self.rasf_ratio:
            main_dict = data_dict['rasf']
            pair_dict = data_dict['rasf_pair']
        else:
            main_dict = data_dict['main']
            pair_dict = data_dict['pair']

        rgb = main_dict['rgb']
        segm = main_dict['segm']
        rgb_pair = pair_dict['rgb']
        segm_pair = pair_dict['segm']
        iter_n = main_dict['iter']

        return_dict = {}

        d_input = torch.cat([rgb, segm, rgb_pair, segm_pair], dim=1)
        if 'return_inp' in data_dict and data_dict['return_inp']:
            return_dict['dpair_inp'] = d_input
            d_input.requires_grad = True

        d_inpdict = dict(input=d_input, iter=iter_n)
        out_dict = self.model(d_inpdict)
        out_dict = dict(dpair_score=out_dict['score'])
        return_dict.update(out_dict)

        return return_dict
