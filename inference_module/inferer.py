import os
import numpy as np
import pydoc
from omegaconf import OmegaConf
import yaml
import munch

import torch
from torch import nn

import utils

from utils.common import dict2device, tti, itt, to_sigm, to_tanh
from utils.common import get_rotation_matrix, segment_img, load_module
from utils import uv_renderer as uvr
from torch.utils.data import DataLoader
from models.efficientnet import EfficientNetLevelEncoder



def get_config(config_path, divide_n_channels=1):        
    with open(config_path, 'r') as f:
        config_ = yaml.load(f, Loader=yaml.FullLoader)

    config = dict()
    for k, v in config_.items():
        if type(v) == dict and 'value' in v:
            config[k] = v['value']
        else:
            config[k] = v

    if 'fix_renderer' not in config:
        config['fix_renderer'] = False
    if 'renderer_checkpoint' not in config:
        config['renderer_checkpoint'] = ""    
    if 'ainp_path' not in config:
        config['ainp_path'] = None
        config['ainp_scales'] = []

    config = munch.munchify(config)

    # config hacks
    config.segm_channels = 1
    config.ainp_path='data/spectral_texture16.pth'
    config.checkpoint512_path = None

    config.bidis_channel_multiplier = int(config.bidis_channel_multiplier / divide_n_channels)
    config.udis_channel_multiplier = int(config.udis_channel_multiplier / divide_n_channels)

    if 'alternating' not in config:
        config.alternating = False

    return config

class Inferer(nn.Module):
    def __init__(self, generator, renderer, generator_config, config):
        super().__init__()

        self.config = config       
        self.generator_config = generator_config
        self.device = config.device
        
        # self.renderer = renderer.eval().to(self.device)
        self.renderer = renderer.eval().to(self.device)
        self.generator = generator.eval().to(self.device)

        self.uv_renderer = uvr.UVRenderer(self.generator_config.image_size, self.generator_config.image_size)
        self.uv_renderer.to(self.device)
        
        self.batch_size = 1
        self.n_rotsamples = 64
        # self.eval()

        # load encoder
        encoder_config_path = os.path.join(config.encoder.experiment_dir, "encoder_config.yaml")
        with open(encoder_config_path) as f:
            encoder_config = OmegaConf.load(f)
        self.encoder = EfficientNetLevelEncoder(
            **utils.common.return_empty_dict_if_none(encoder_config.model.encoder.args)
        )
        state_dict = torch.load(os.path.join(config.encoder.experiment_dir, "checkpoints", config.encoder.checkpoint_name))
        self.encoder.load_state_dict(state_dict['inferer']['encoder'])
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        # trainable parameters
        ## latent
        latent_mean, _ = self.calc_latent_stats(
            n_mean_latent=config.inferer.n_mean_latent
        )

        latent_init = latent_mean \
                     .detach() \
                     .clone() \
                     .view(1, 1, self.generator.style_dim) \
                     .repeat(1, self.generator.n_latent, 1)

        self.latent = nn.Parameter(latent_init, requires_grad=True)

        ## noise
        noise_init = self.generator.make_noise()
        self.noises = nn.ParameterList([nn.Parameter(x, requires_grad=True) for x in noise_init])

        ## ntexture
        self.ntexture = None

        # diff renderer
        self.diff_uv_renderer = utils.uv_renderer.NVDiffRastUVRenderer(config.inferer.faces_path, config.inferer.uv_vert_values_path)

        
    def sample_ntexture(self, ntexture, uv):
        return torch.nn.functional.grid_sample(ntexture, uv.permute(0, 2, 3, 1), align_corners=True)

    def infer_pass(self, noise, verts, ntexture=None, uv=None, sampled_ntexture=None):

        if verts is not None:
            B = verts.shape[0]
        else:
            B = uv.shape[0]


        if uv is None and sampled_ntexture is None:
            uv = self.uv_renderer(verts)

        if type(noise) == torch.Tensor and noise.ndim == 3:
            noise = torch.split(noise, 1, dim=1)
            noise = [x[0] for x in noise]

        if ntexture is None and sampled_ntexture is None:
            fake_ntexture = self.generator(noise, noise=self.noises, input_is_latent=True)
            fake_ntexture = torch.cat([fake_ntexture]*B, dim=0)
        else:
            fake_ntexture = ntexture

        if sampled_ntexture is None:
            sampled_ntexture = self.sample_ntexture(fake_ntexture, uv)

        renderer_dict_out = self.renderer(dict(uv=uv, sampled_ntextures=sampled_ntexture, nrender=sampled_ntexture))
        fake_img = renderer_dict_out['fake_rgb']
        fake_segm = renderer_dict_out['fake_segm']
        fake_img = segment_img(fake_img, fake_segm)

        out = dict(
            fake_img=fake_img,
            fake_segm=fake_segm,
            fake_ntexture=fake_ntexture,
            uv=uv,
            sampled_ntexture=sampled_ntexture
        )

        return out

    def calc_latent_stats(self, n_mean_latent=10000):
        with torch.no_grad():
            noise_sample = torch.randn(n_mean_latent, self.generator.style_dim).to(self.device)
            latent_out = self.generator.style(noise_sample)
            latent_mean = latent_out.mean(0)
            latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

        return latent_mean, latent_std

    def make_noise_multiple(self, batch_size, latent_dim, n_noises):
        noises = torch.randn(n_noises, batch_size, latent_dim, device=self.device).unbind(0)
        return noises

    def get_state_dict(self):
        # collect generator parameters
        generator_params = []
        generator_params.extend(self.generator.conv1.parameters())
        for l in self.generator.convs:
            generator_params.extend(l.parameters())  
        generator_params.extend(self.generator.to_rgb1.parameters())
        for l in self.generator.to_rgbs:
            generator_params.extend(l.parameters())
            
        state_dict = {
            'latent': self.latent,
            'noise': self.noise,
            'generator_params': generator_params,
            'ntexture': self.ntexture
        }

        return state_dict