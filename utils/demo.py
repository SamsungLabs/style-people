import pickle
import numpy as np
import smplx
import torch
import importlib

import models
from models.generator import Generator
from models.renderer import Renderer
from utils.bbox import get_ltrb_bbox, crop_resize_verts
from utils.common import get_rotation_matrix, rotate_verts, to_sigm
from utils.common import dict2device, setup_environment
from utils.uv_renderer import UVRenderer

import os
import collections
import pydoc

import hydra
from omegaconf import OmegaConf 
import yaml

import inference_module
from dataloaders.inference_loader import ExampleDataset

from inference_module.inferer import load_models, get_config


class DemoInferer():
    def __init__(self, checkpoint_path, smplx_model_path, imsize=1024, device='cuda:0'):
        self.smplx_model = smplx.body_models.SMPLX(smplx_model_path).to(device)
        
        self.checkpoint_path = checkpoint_path
        experiment_dir = '/'.join(self.checkpoint_path.split('/')[:-2])
        checkpoint_file = self.checkpoint_path.split('/')[-1]
        
        self.generator_config = get_config(experiment_dir)
        self.models = load_models(experiment_dir, self.generator_config, checkpoint_file)

        self.generator, self.renderer = self.models['generator'], self.models['renderer']
        self.v_inds = torch.LongTensor(np.load('data/v_inds.npy')).to(device)
        self.input_size = imsize // 2  # input resolution is twice as small as output

        self.uv_renderer = UVRenderer(self.input_size, self.input_size).to(device)

        self.device = device
        self.style_dim = 512
        
    def infer(self, config, input_path, train_frames, val_frames, save_dir):
        # setup environment
        setup_environment(config.random_seed)

        dataloader = ExampleDataset(root_dir=input_path, image_h=self.input_size, image_w=self.input_size)
        train_dict = dataloader.load(frame_indices=list(map(int, train_frames.split(','))))
        train_dict = dict2device(train_dict, self.device, dtype=torch.float32)

        val_dict = dataloader.load(frame_indices=list(map(int, val_frames.split(','))))
        val_dict = dict2device(val_dict, self.device, dtype=torch.float32)

        print("Successfully loaded data")

        # load inferer
        inferer = inference_module.inferer.Inferer(self.models, self.generator_config, config)
        inferer = inferer.to(self.device)
        inferer.eval()
        print("Successfully loaded inferer")

        # load runner
        runner = inference_module.runner.Runner(config, inferer)
        print("Successfully loaded runner")

        # train loop
        runner.run_epoch(train_dict, val_dict, save_dir=save_dir)

    def sample_texture(self):
        z_val = [models.styleganv2.modules.make_noise(1, self.style_dim, 1, self.device)]
        ntexture = self.generator(z_val)['ntexture']
        return ntexture

    def load_smplx(self, sample_path):
        with open(sample_path, 'rb') as f:
            smpl_params = pickle.load(f)

        for k, v in smpl_params.items():
            smpl_params[k] = torch.FloatTensor(v).to(self.device)

        smpl_output = self.smplx_model(**smpl_params)
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

        return rgb_frames
