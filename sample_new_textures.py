import argparse
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from utils.common import tti, set_random_seed
from utils.demo import DemoInferer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--textures_root', type=str, default='data/textures')
    parser.add_argument('--checkpoint_path', type=str, default='data/checkpoint.pth')
    parser.add_argument('--smplx_model_path', type=str, default='data/smplx/SMPLX_NEUTRAL.pkl')
    parser.add_argument('--texture_batch_name', type=str)
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--n_rotimgs', type=int, default=8)
    parser.add_argument('--imsize', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    set_random_seed(args.seed)

    inferer = DemoInferer(args.checkpoint_path, args.smplx_model_path, imsize=args.imsize, device=args.device)

    for i in tqdm(range(args.n_samples)):
        with torch.no_grad():
            ntexture = inferer.sample_texture()

        texture_save_dir = os.path.join(args.textures_root, args.texture_batch_name, f"{i:04}")
        os.makedirs(texture_save_dir, exist_ok=True)

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
