from models.generator import Generator
import pickle

import numpy as np
import smplx
import torch

import models
from models.generator import Generator
from models.renderer import Renderer
from utils.bbox import get_ltrb_bbox, crop_resize_verts
from utils.common import get_rotation_matrix, to_sigm
from utils.uv_renderer import UVRenderer


def rotate_verts(vertices, angle, K, K_inv, axis='y', mean_point=None):
    rot_mat = get_rotation_matrix(angle, axis)
    rot_mat = torch.FloatTensor(rot_mat).to(vertices.device).unsqueeze(0)

    vertices_world = torch.bmm(vertices, K_inv.transpose(1, 2))
    if mean_point is None:
        mean_point = vertices_world.mean(dim=1)

    vertices_rot = vertices_world - mean_point
    vertices_rot = torch.bmm(vertices_rot, rot_mat.transpose(1, 2))
    vertices_rot = vertices_rot + mean_point
    vertices_rot_cam = torch.bmm(vertices_rot, K.transpose(1, 2))

    return vertices_rot_cam, mean_point


def load_models(checkpoint_path='data/checkpoint.pth', device='cuda:0'):
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

    def __init__(self, checkpoint_path, smplx_model_path, imsize=1024, device='cuda:0'):
        self.smplx_model = smplx.body_models.SMPLX(smplx_model_path).to(device)
        self.generator, self.renderer = load_models(checkpoint_path)
        self.v_inds = torch.LongTensor(np.load('data/v_inds.npy')).to(device)
        self.input_size = imsize // 2  # input resolution is twice as small as output

        self.uv_renderer = UVRenderer(self.input_size, self.input_size).to(device)

        self.device = device
        self.style_dim = 512

    def sample_texture(self):
        z_val = [models.styleganv2.modules.make_noise(1, self.style_dim, 1, self.device)]
        ntexture = self.generator(z_val)
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
