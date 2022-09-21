import pickle
import numpy as np
import smplx
import torch
import importlib
from torch.utils.data import DataLoader
from collections import defaultdict

import models
from models.generator import Generator
from models.renderer import Renderer
from utils.bbox import get_ltrb_bbox, crop_resize_verts
from utils.common import get_rotation_matrix, rotate_verts, to_sigm
from utils.common import dict2device, setup_environment
from utils.uv_renderer import UVRenderer
from utils.smplx_models import build_smplx_model_dict

import os
import collections

import inference_module
from dataloaders.inference_loader import InferenceDataset

from inference_module.inferer import get_config


def concat_all_samples(dataloader):
    datadicts = defaultdict(list)

    for data_dict in dataloader:
        for k, v in data_dict.items():
            datadicts[k].append(v)

    dict_combined = {}
    for k, v in datadicts.items():
        if type(v[0]) == torch.Tensor:
            dict_combined[k] = torch.cat(v, dim=0)
        elif type(v[0]) == list:
            dict_combined[k] = [x[0] for x in v]

    return dict_combined

def load_models(checkpoint_path='data/checkpoints/generative_model.pth', device='cuda:0'):
    ainp_path = 'data/spectral_texture16.pth'
    ainp_scales = [64, 128, 256, 512]

    ainp_tensor = torch.load(ainp_path)
    generator = Generator(ainp_tensor=ainp_tensor, ainp_scales=ainp_scales).to(device)
    renderer = Renderer().to(device)

    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint['g'])
    renderer.load_state_dict(checkpoint['r'])

    generator.eval()
    renderer.eval()

    return generator, renderer




class DemoInferer():
    def __init__(self, checkpoint_path, smplx_model_dir, imsize=1024, config_path='data/config.yaml', device='cuda:0'):

        self.smplx_model_dir = smplx_model_dir
        self.smplx_models_dict = build_smplx_model_dict(smplx_model_dir, device)
        
        self.generator_config = get_config(config_path)
        self.generator, self.renderer = load_models(checkpoint_path, device) 

        self.v_inds = torch.LongTensor(np.load('data/v_inds.npy')).to(device)


        self.image_size = imsize
        self.input_size = imsize // 2  # input resolution is twice as small as output

        self.uv_renderer = UVRenderer(self.input_size, self.input_size).to(device)

        self.device = device
        self.style_dim = 512
        
    def infer(self, config, input_path):
        # setup environment
        setup_environment(config.random_seed)
        dataset = InferenceDataset(input_path, self.image_size, self.v_inds, self.smplx_model_dir)
        dataloader = DataLoader(dataset)
        train_dict = concat_all_samples(dataloader)
        train_dict = dict2device(train_dict, self.device)

        print("Successfully loaded data")

        # load inferer
        inferer = inference_module.inferer.Inferer(self.generator, self.renderer, self.generator_config, config)
        inferer = inferer.to(self.device)
        inferer.eval()
        print("Successfully loaded inferer")

        # load runner
        runner = inference_module.runner.Runner(config, inferer, self.smplx_models_dict, self.image_size)
        print("Successfully loaded runner")

        # train loop
        ntexture = runner.run_epoch(train_dict)

        return ntexture

    def sample_texture(self):
        z_val = [models.styleganv2.modules.make_noise(1, self.style_dim, 1, self.device)]
        ntexture = self.generator(z_val)
        return ntexture

    def load_smplx(self, sample_path):
        with open(sample_path, 'rb') as f:
            smpl_params = pickle.load(f)

        gender = smpl_params['gender']

        for k, v in smpl_params.items():
            if type(v) == np.ndarray:
                if 'hand_pose' in k:
                    v = v[:, :6]
                smpl_params[k] = torch.FloatTensor(v).to(self.device)

        smpl_output = self.smplx_models_dict[gender](**smpl_params)
        vertices = smpl_output.vertices
        vertices = vertices[:, self.v_inds]
        K = smpl_params['camera_intrinsics'].unsqueeze(0)
        vertices = torch.bmm(vertices, K.transpose(1, 2))
        return vertices, K

    def crop_vertices(self, vertices, K):
        ltrb = get_ltrb_bbox(vertices)
        vertices, K = crop_resize_verts(vertices, K, ltrb, self.input_size)
        return vertices, K, ltrb

    def make_rgb(self, vertices, ntexture):
        uv = self.uv_renderer(vertices, negbg=True)

        nrender = torch.nn.functional.grid_sample(ntexture, uv.permute(0, 2, 3, 1), align_corners=True)
        renderer_input = dict(uv=uv, nrender=nrender)

        with torch.no_grad():
            renderer_output = self.renderer(renderer_input)

        fake_rgb = renderer_output['fake_rgb']
        fake_segm = renderer_output['fake_segm']
        fake_rgb = to_sigm(fake_rgb) * (fake_segm > 0.8)

        return fake_rgb

    def make_rotation_images(self, ntexture, n_rotimgs, smplx_path='data/smplx_sample.pkl'):
        vertices, K = self.load_smplx(smplx_path)
        vertices, K, ltrb = self.crop_vertices(vertices, K)

        K_inv = torch.inverse(K)

        rgb_frames = []
        for j in range(n_rotimgs):
            angle = np.pi * 2 * j / n_rotimgs
            verts_rot, mean_point = rotate_verts(vertices, angle, K, K_inv, axis='y')
            rgb = self.make_rgb(verts_rot, ntexture)
            rgb_frames.append(rgb)

        return rgb_frames, ltrb
