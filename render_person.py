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
    parser.add_argument('--checkpoint_path', type=str, default='data/checkpoint.pth')
    parser.add_argument('--smplx_model_path', type=str, default='data/smplx/SMPLX_NEUTRAL.pkl')
    parser.add_argument('--texture_path', type=str)
    parser.add_argument('--smplx_dict_path', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--imsize', type=int, default=1024)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    inferer = DemoInferer(args.checkpoint_path, args.smplx_model_path, imsize=args.imsize, device=args.device)
    ntexture = torch.load(args.texture_path).to(args.device)

    vertices, K = inferer.load_smplx(args.smplx_dict_path)
    vertices, K, ltrb = inferer.crop_vertices(vertices, K)
    rgb = inferer.make_rgb(vertices, ntexture)
    rgb = (tti(rgb) * 255).astype(np.uint8)

    rgb_out_path = os.path.join(args.save_dir, f"rgb.png")
    cv2.imwrite(rgb_out_path, rgb[..., ::-1])

    ltrb = ltrb[0].cpu().numpy().tolist()
    with open(os.path.join(args.save_dir, f"ltrb.json"), 'w') as f:
        json.dump(ltrb, f)
