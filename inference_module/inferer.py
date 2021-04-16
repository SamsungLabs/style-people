import os
from tqdm import tqdm
import numpy as np
import pydoc
import random
import pickle
import copy
from omegaconf import OmegaConf
import yaml
import munch
import importlib

import torch
from torch import nn

import smplx
import utils
import inference_module

from utils.common import dict2device, tti, itt, to_sigm, to_tanh
from utils.common import get_rotation_matrix, segment_img, load_module
from utils import uv_renderer as uvr
from torch.utils.data import DataLoader


NUM_PCA_COMPONENTS = 12


def get_config(experiment_dir, divide_n_channels=1):        
    config_path = os.path.join(experiment_dir, 'config.yaml')
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


def get_cpfile(experiment_dir, checkpoint_id):
    cps = sorted(os.listdir(os.path.join(experiment_dir, 'checkpoints')))
    cp = cps[checkpoint_id]
    return cp


def load_models(experiment_dir, config, cp_file, nointerdirs=False):
    m_generator = load_module('models', 'generator').Wrapper
    m_renderer = load_module('models', 'renderer').Wrapper

    generator = m_generator.get_net(config)
    renderer = m_renderer.get_net(config)
    g_ema = m_generator.get_net(config)

    if nointerdirs:
        cp_path = os.path.join(experiment_dir, cp_file)
    else:
        cp_path = os.path.join(experiment_dir, 'checkpoints', cp_file)
    state_dict = torch.load(cp_path, map_location='cpu')
    
    generator.load_state_dict(state_dict['g'])

    if 'r_ema' in state_dict:
        renderer.load_state_dict(state_dict['r_ema'])
#         print("Loaded renderer weights from r_ema")
    else:
        renderer.load_state_dict(state_dict['r'])

    g_ema.load_state_dict(state_dict['g_ema'])

    models = dict(renderer=renderer, generator=generator, g_ema=g_ema)
    return models


class Inferer(nn.Module):
    def __init__(self, models, generator_config, config):
        super().__init__()

        self.config = config       
        self.generator_config = generator_config
        self.device = config.device
        
        self.renderer = models['renderer'].eval().to(self.device)
        self.generator = models['generator'].eval().to(self.device)
        self.g_ema = models['g_ema'].eval().to(self.device)

        self.uv_renderer = uvr.UVRenderer(self.generator_config.image_size, self.generator_config.image_size)
        self.uv_renderer.to(self.device)
        
        self.batch_size = 1
        self.n_rotsamples = 64
        self.eval()

        # load discriminators
        cp_path = os.path.join(config.generator.experiment_dir, 'checkpoints', config.generator.checkpoint_name)
        state_dict = torch.load(cp_path, map_location='cpu')

        dicriminator_names = generator_config.discriminator_list.split(', ')

        self.discriminators = dict()
        for i, (dicriminator_name, dicriminator_new_name) in enumerate(zip(dicriminator_names, ('unary', 'binary', 'face'))):
            m_dicriminator = load_module('models.discriminators', dicriminator_name).Wrapper
            dicriminator = m_dicriminator.get_net(generator_config)
            dicriminator.load_state_dict(state_dict['d'][i])
            
            self.discriminators[dicriminator_new_name] = dicriminator
            print(f"Loaded discriminator: {dicriminator_name}")

        if 'face' not in self.discriminators:
            # load face dicriminator from other checkpoint
            face_generator_config = get_config(config.generator.face_experiment_dir, divide_n_channels=config.generator.face_divide_n_channels)

            cp_path = os.path.join(config.generator.face_experiment_dir, 'checkpoints', config.generator.face_checkpoint_name)
            state_dict = torch.load(cp_path, map_location='cpu')

            dicriminator_names = face_generator_config.discriminator_list.split(', ')
            dicriminator_name = dicriminator_names[-1]
            
            m_dicriminator = utils.utils.load_module('discriminators', dicriminator_name).Wrapper
            dicriminator = m_dicriminator.get_net(face_generator_config)
            dicriminator.load_state_dict(state_dict['d'][-1])
            
            self.discriminators['face'] = dicriminator

            print("Loaded face discriminator from other checkpoint")

        # load encoder
        encoder_config_path = os.path.join(config.encoder.experiment_dir, "encoder_config.yaml")
        with open(encoder_config_path) as f:
            encoder_config = OmegaConf.load(f)
        encoder_cls = pydoc.locate('models.efficientnet.EfficientNetLevelEncoder')
        self.encoder = encoder_cls(
            **utils.common.return_empty_dict_if_none(encoder_config.model.encoder.args)
        )
        state_dict = torch.load(os.path.join(config.encoder.experiment_dir, "checkpoints", config.encoder.checkpoint_name))
        self.encoder.load_state_dict(state_dict['inferer']['encoder'])
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        # trainable parameters
        ## latent
        self.latent_mean, self.latent_std = self.calc_latent_stats(
            n_mean_latent=config.inferer.n_mean_latent
        )

        latent_init = self.latent_mean \
                     .detach() \
                     .clone() \
                     .view(1, 1, self.g_ema.style_dim) \
                     .repeat(1, self.g_ema.n_latent, 1)

        self.latent = nn.Parameter(latent_init, requires_grad=True)

        ## noise
        noise_init = self.g_ema.make_noise()
        self.noise = nn.ParameterList([nn.Parameter(x, requires_grad=True) for x in noise_init])

        ## ntexture
        self.ntexture = None

        # diff renderer
        self.diff_uv_renderer = utils.uv_renderer.NVDiffRastUVRenderer(config.inferer.faces_path, config.inferer.uv_vert_values_path)

        # smplx
        smplx_models = dict()
        for gender in ['female', 'male']:
            model = smplx.create(
                config.inferer.body_models_path,
                model_type='smplx',
                gender=gender,
                batch_size=1,
                create_transl=False,
                num_pca_comps=NUM_PCA_COMPONENTS
            )
    
            model.eval()
            smplx_models[gender] = model

        self.smplx_models = nn.ModuleDict(smplx_models)       
        
    def sample_ntexture(self, ntexture, uv):
        return torch.nn.functional.grid_sample(ntexture, uv.permute(0, 2, 3, 1), align_corners=True)

    def infer_pass(self, noise, verts, noises=None, ntexture=None, uv=None, sampled_ntexture=None, return_latents=False, input_is_latent=False, infer_g_ema=False):
        if uv is None and sampled_ntexture is None:
            uv = self.uv_renderer(verts)

        if ntexture is None and sampled_ntexture is None:
            if infer_g_ema:
                g_out = self.g_ema(noise, noise=noises, return_latents=return_latents, input_is_latent=input_is_latent, )
            else:
                g_out = self.generator(noise, noise=noises, return_latents=return_latents, input_is_latent=input_is_latent, )

            fake_ntexture = g_out['fake_rgb']
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
        if return_latents:
            out['w'] = g_out['latent']

        return out

    def calc_latent_stats(self, n_mean_latent=10000):
        with torch.no_grad():
            noise_sample = torch.randn(n_mean_latent, self.g_ema.style_dim).to(self.device)
            latent_out = self.g_ema.style(noise_sample)
            latent_mean = latent_out.mean(0)
            latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

        return latent_mean, latent_std

    def make_noise_multiple(self, batch_size, latent_dim, n_noises):
        noises = torch.randn(n_noises, batch_size, latent_dim, device=self.device).unbind(0)
        return noises

    def get_state_dict(self):
        # collect generator parameters
        generator_params = []
        generator_params.extend(self.g_ema.conv1.parameters())
        for l in self.g_ema.convs:
            generator_params.extend(l.parameters())  
        generator_params.extend(self.g_ema.to_rgb1.parameters())
        for l in self.g_ema.to_rgbs:
            generator_params.extend(l.parameters())
            
        state_dict = {
            'latent': self.latent,
            'noise': self.noise,
            'generator_params': generator_params,
            'ntexture': self.ntexture
        }

        return state_dict