import numpy as np
import minimal_pytorch_rasterizer
import torch
from torch import nn


class UVRenderer(torch.nn.Module):
    def __init__(self, H, W, faces_path='data/uv_renderer/face_tex.npy',
                 vertice_values_path='data/uv_renderer/uv.npy'):
        super().__init__()
        faces_cpu = np.load(faces_path)
        uv_cpu = np.load(vertice_values_path)

        self.faces = torch.nn.Parameter(torch.tensor(faces_cpu, dtype=torch.int32).contiguous(), requires_grad=False)
        self.vertice_values = torch.nn.Parameter(torch.tensor(uv_cpu, dtype=torch.float32).contiguous(),
                                                 requires_grad=False)

        self.pinhole = minimal_pytorch_rasterizer.Pinhole2D(
            fx=1, fy=1,
            cx=0, cy=0,
            h=H, w=W
        )

    def set_vertice_values(self, vertive_values):
        self.vertice_values = torch.nn.Parameter(
            torch.tensor(vertive_values, dtype=torch.float32).to(self.vertice_values.device), requires_grad=False)

    def forward(self, verts, norm=True, negbg=True, return_mask=False):
        N = verts.shape[0]

        uvs = []
        for i in range(N):
            v = verts[i]
            uv = minimal_pytorch_rasterizer.project_mesh(v, self.faces, self.vertice_values, self.pinhole)
            uvs.append(uv)

        uvs = torch.stack(uvs, dim=0).permute(0, 3, 1, 2)
        mask = (uvs > 0).sum(dim=1, keepdim=True).float().clamp(0., 1.)

        if norm:
            uvs = (uvs * 2 - 1.)

        if negbg:
            uvs = uvs * mask - 10 * torch.logical_not(mask)

        if return_mask:
            return uvs, mask
        else:
            return uvs
