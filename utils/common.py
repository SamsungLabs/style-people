import numpy as np
import random
import os
import time
import importlib

import cv2
from PIL import Image

import math
import pickle

import torch
from torch import distributed as dist
from torch.utils.data.sampler import Sampler


def load_module(module_type, module_name):
    m = importlib.import_module(f'{module_type}.{module_name}')
    return m


def return_empty_dict_if_none(x):
    return {} if x is None else x


def get_data_sampler(dataset, shuffle=False, is_distributed=False):
    if is_distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return torch.utils.data.RandomSampler(dataset)
    else:
        return torch.utils.data.SequentialSampler(dataset)


def dict2device(d, device, dtype=None):
    if isinstance(d, np.ndarray):
        d = torch.from_numpy(d)

    if torch.is_tensor(d):
        d = d.to(device)
        if dtype is not None:
            d = d.type(dtype)
        return d

    if isinstance(d, dict):
        for k, v in d.items():
            d[k] = dict2device(v, device, dtype=dtype)

    return d


def setup_environment(seed):
    # random
    random.seed(seed)

    # numpy
    np.random.seed(seed)
    
    # cv2
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    # pytorch
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


def squeeze_metrics(d):
    metrics = dict()
    for k, v in d.items():
        if torch.is_tensor(v):
            metrics[k] = v.mean().item()
        elif isinstance(v, float):
            metrics[k] = v
        else:
            raise NotImplementedError("Unknown datatype for metric: {}".format(type(v)))

    return metrics


def reduce_metrics(metrics):
    metrics_dict = dict()
    for k in metrics[0].keys():
        metrics_dict[k] = np.mean([item[k] for item in metrics])

    return metrics_dict


def get_world_size():
    if not dist.is_available():
        return 1

    if not dist.is_initialized():
        return 1

    return dist.get_world_size()


def reduce_loss_dict(loss_dict):
    world_size = get_world_size()

    if world_size < 2:
        return loss_dict

    with torch.no_grad():
        keys = []
        losses = []

        for k in sorted(loss_dict.keys()):
            keys.append(k)
            losses.append(loss_dict[k])

        losses = torch.stack(losses, 0)
        dist.reduce(losses, dst=0)

        if dist.get_rank() == 0:
            losses /= world_size

        reduced_losses = {k: v for k, v in zip(keys, losses)}

    return reduced_losses


def flatten_parameters(parameters):
    list_of_flat_parameters = [torch.flatten(p) for p in parameters]
    flat_parameters = torch.cat(list_of_flat_parameters).view(-1, 1)
    return flat_parameters


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def itt(img):
    tensor = torch.FloatTensor(img)  #
    if len(tensor.shape) == 3:
        tensor = tensor.permute(2, 0, 1)
    else:
        tensor = tensor.unsqueeze(0)
    return tensor


def tti(tensor):
    tensor = tensor.detach().cpu()
    tensor = tensor[0].permute(1, 2, 0)
    image = tensor.numpy()
    if image.shape[-1] == 1:
        image = image[..., 0]
    return image


def to_tanh(t):
    return t * 2 - 1.


def to_sigm(t):
    return (t + 1) / 2


def get_rotation_matrix(angle, axis='x'):
    if axis == 'x':
        return np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
    elif axis == 'y':
        return np.array([
            [np.cos(angle), 0, -np.sin(angle)],
            [0, 1, 0],
            [np.sin(angle), 0, np.cos(angle)]
        ])
    elif axis == 'z':
        return np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError(f"Unkown axis {axis}")
        
        
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


def json2kps(openpose_dict):
    list2kps = lambda x: np.array(x).reshape(-1, 3)
    keys_to_save = ['pose_keypoints_2d', 'face_keypoints_2d', 'hand_right_keypoints_2d', 'hand_left_keypoints_2d']

    kps = openpose_dict['people']
    if len(kps) == 0:
        kp_stacked = np.ones((137, 2)) * -1
        return kp_stacked
    kps = kps[0]
    kp_parts = [list2kps(kps[key]) for key in keys_to_save]
    kp_stacked = np.concatenate(kp_parts, axis=0)
    kp_stacked[kp_stacked[:, 2] < 0.1, :] = -1
    kp_stacked = kp_stacked[:, :2]

    return kp_stacked


def segment_img(img, segm):
    img = to_sigm(img) * segm
    img = to_tanh(img)
    return img


def segm2mask(segm):
    segm = torch.sum(segm, dim=1, keepdims=True)  # Bx3xHxW -> Bx1xHxW
    segm = (segm > 0.0).type(torch.float32)
    return segm