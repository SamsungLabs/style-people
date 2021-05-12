import argparse
import os
import torch
import cv2
import numpy as np

from utils.common import tti
from utils.demo import DemoInferer
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description='')
<<<<<<< HEAD
    parser.add_argument('--checkpoint_path', type=str, default='data/checkpoints/generative_model.pth', help='Path to generative model checkpoint')
=======
    parser.add_argument('--checkpoint_path', type=str, default='data/checkpoint/generative_model.pth', help='Path to generative model checkpoint')
>>>>>>> master
    parser.add_argument('--config_path', type=str, default='inference_module/config.yaml')
    parser.add_argument('--smplx_model_dir', type=str, default='data/smplx/', help='Path to smplx models')
    parser.add_argument('--input_path', type=str, default='data/inference_samples/azure_02', help='Path to a directory that contains data samples')
    parser.add_argument('--texture_out_dir', type=str, default='data/textures/azure_02', help='Path to a directory to save fitted texture in')
    parser.add_argument('--n_rotimgs', type=int, default=8, help='Number of rotation steps to render textured model in')
    parser.add_argument('--imsize', type=int, default=1024, help='Resolution in which to render rotation steps')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run inference process on')
    args = parser.parse_args()

    texture_save_dir = args.texture_out_dir
    os.makedirs(texture_save_dir, exist_ok=True)


    inferer = DemoInferer(args.checkpoint_path, args.smplx_model_dir, imsize=args.imsize, device=args.device)
    config = OmegaConf.load(args.config_path)
    ntexture = inferer.infer(config, args.input_path)

    texture_out_path = os.path.join(texture_save_dir, 'texture.pth')
    torch.save(ntexture.cpu(), texture_out_path)
    if args.n_rotimgs > 0:
        rot_images = inferer.make_rotation_images(ntexture, args.n_rotimgs)

        for j, rgb in enumerate(rot_images):
            rgb = tti(rgb)
            rgb = (rgb * 255).astype(np.uint8)

            rgb_out_path = os.path.join(texture_save_dir, 'rotation_images', f"{j:04d}.png")
            os.makedirs(os.path.dirname(rgb_out_path), exist_ok=True)
            cv2.imwrite(rgb_out_path, rgb[..., ::-1])


if __name__ == '__main__':
    main()
