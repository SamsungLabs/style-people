import argparse
import json
import os

import cv2
import numpy as np
import torch

from utils.common import tti
from utils.demo import DemoInferer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--checkpoint_path', type=str, default='data/checkpoint/generative_model.pth', help='Path to generative model checkpoint')
    parser.add_argument('--smplx_model_dir', type=str, default='data/smplx/', help='Path to smplx models')
    parser.add_argument('--texture_path', type=str, help='Path to a .pth neural texture file')
    parser.add_argument('--smplx_dict_path', type=str, help='Path to a .pkl file with smplx parameters')
    parser.add_argument('--save_dir', type=str, help='Path to a directory to save generated images in')
    parser.add_argument('--n_rotimgs', type=int, default=8, help='Number of rotation steps to render textured model in')
    parser.add_argument('--imsize', type=int, default=1024, help='Resolution in which to render images (1024 recommended)')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run images generation process on')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    inferer = DemoInferer(args.checkpoint_path, args.smplx_model_dir, imsize=args.imsize, device=args.device)
    ntexture = torch.load(args.texture_path).to(args.device)

    # vertices, K = inferer.load_smplx(args.smplx_dict_path)
    # vertices, K, ltrb = inferer.crop_vertices(vertices, K)
    # rgb = inferer.make_rgb(vertices, ntexture)
    # rgb = (tti(rgb) * 255).astype(np.uint8)

    # rgb_out_path = os.path.join(args.save_dir, f"rgb.png")
    # cv2.imwrite(rgb_out_path, rgb[..., ::-1])

    rot_images, ltrb = inferer.make_rotation_images(ntexture, args.n_rotimgs, smplx_path=args.smplx_dict_path)


    for j, rgb in enumerate(rot_images):
        rgb = tti(rgb)
        rgb = (rgb * 255).astype(np.uint8)

        if j == 0:
            rgb_out_path = os.path.join(args.save_dir, f"rgb.png")
            os.makedirs(os.path.dirname(rgb_out_path), exist_ok=True)
            cv2.imwrite(rgb_out_path, rgb[..., ::-1])    

        rgb_out_path = os.path.join(args.save_dir, 'rotation_images', f"{j:04d}.png")
        os.makedirs(os.path.dirname(rgb_out_path), exist_ok=True)
        cv2.imwrite(rgb_out_path, rgb[..., ::-1])    

    ltrb = ltrb[0].cpu().numpy().tolist()
    with open(os.path.join(args.save_dir, f"ltrb.json"), 'w') as f:
        json.dump(ltrb, f)
