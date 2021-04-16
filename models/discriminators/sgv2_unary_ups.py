import math

import torch
from torch import nn, optim
from models.discriminators import styleganv2_ups

class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--image_size', type=int, default=256)
        parser.add('--target_size', type=int)
        parser.add('--udis_input_channels', type=int, default=4)
        parser.add('--udis_channel_multiplier', type=int, default=6)
        parser.add('--lr_dis', type=float, default=0.002)
        parser.add('--fadein_steps', type=float, default=5000)
        parser.add('--fadestep_every', type=float, default=10)

    @staticmethod
    def get_net(args):
        if not hasattr(args, 'target_size') or args.target_size is None:
            args.target_size = args.image_size

        net = Discriminator(args.target_size, input_channels=args.udis_input_channels, 
            channel_multiplier=args.udis_channel_multiplier, alternating=args.alternating, 
            fadein_steps=args.fadein_steps, fadestep_every=args.fadestep_every).to(args.device)
        return net

    @staticmethod
    def get_optimizer(discriminator, args):
        d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
        optimizer = optim.Adam(
            discriminator.parameters(),
            lr=args.lr_dis * d_reg_ratio,
            betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
        )

        return dict(d_optim=optimizer)


class Discriminator(nn.Module):
    def __init__(self, size, input_channels=3, channel_multiplier=4, alternating=False,
                                        fadein_steps=5000, fadestep_every=10):
        super().__init__()

        self.model = styleganv2_ups.Discriminator(size, input_channels, channel_multiplier, 
            fadein_steps=fadein_steps, fadestep_every=fadestep_every)
        self.alternating = alternating

    def load_prev_scale(self, state_dict):
        state_dict = {k[6:]:v for k,v in state_dict.items()}
        self.model.load_prev_scale(state_dict)
        print('UNARY LOADED')

    def forward(self, data_dict):
        if self.alternating and 'pair' in data_dict:
            return dict()
            
        data_dict_ = data_dict['main']
        rgb = data_dict_['rgb']
        segm = data_dict_['segm']
        iter_n = data_dict_['iter']
        return_dict = {}

        d_input = torch.cat([rgb, segm], dim=1)


        if 'return_inp' in data_dict and data_dict['return_inp']:
            return_dict['d_inp'] = d_input#.detach()
            d_input.requires_grad = True

        d_inpdict = dict(input=d_input, iter=iter_n)
        out_dict = self.model(d_inpdict)
        out_dict = dict(d_score=out_dict['score'])
        return_dict.update(out_dict)

        return return_dict
