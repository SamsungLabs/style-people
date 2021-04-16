import argparse
import json
import os

from utils.demo import DemoInferer
from omegaconf import OmegaConf 


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--checkpoint_path', type=str, default='data/checkpoints/generator.pth.tar')
    parser.add_argument('--config_path', type=str, default='inference_module/config.yaml')
    parser.add_argument('--smplx_model_path', type=str, default='data/smplx/SMPLX_NEUTRAL.pkl')
    parser.add_argument('--input_path', type=str, default='data/example_input/female-1-casual')
    parser.add_argument('--train_frames', type=str, default='13')
    parser.add_argument('--val_frames', type=str, default='73')
    parser.add_argument('--save_dir', type=str, default='data/output/inference')
    parser.add_argument('--imsize', type=int, default=1024)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    
    inferer = DemoInferer(args.checkpoint_path, args.smplx_model_path, imsize=args.imsize, device=args.device)
    config = OmegaConf.load(args.config_path)  
    inferer.infer(config, args.input_path, args.train_frames, args.val_frames, args.save_dir)

    
if __name__ == '__main__':
    main()
    