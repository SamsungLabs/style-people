import torch
import numpy as np
import random


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


def dict2device(d, device):
    if type(d) == torch.Tensor:
        return d.to(device)

    if type(d) == dict:
        for k, v in d.items():
            d[k] = dict2device(v, device)
    return d


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