import numpy as np
import minimal_pytorch_rasterizer
import torch
from torch import nn

import nvdiffrast.torch as dr


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

        
class NVDiffRastUVRenderer(torch.nn.Module):
    def __init__(self, faces_path='data/uv_renderer/face_tex.npy',
                 uv_vert_values_path='data/uv_renderer/uv.npy'):
        super().__init__()
        
        self.glctx = dr.RasterizeGLContext()
        
        # load faces
        self.faces = nn.Parameter(
            torch.tensor(np.load(faces_path), dtype=torch.int32).contiguous(),
            requires_grad=False
        )
        
        # load uv vert values
        self.uv_vert_values = nn.Parameter(
            torch.tensor(np.load(uv_vert_values_path), dtype=torch.float32).contiguous(),
            requires_grad=False
        )
        
    def convert_to_ndc(self, verts, calibration_matrix, orig_w, orig_h, near=0.0001, far=10.0, invert_verts=True):
        device = verts.device
        
        # unproject verts
        if invert_verts:
            calibration_matrix_inv = torch.inverse(calibration_matrix)
            verts_3d = torch.bmm(verts, calibration_matrix_inv.transpose(1, 2))
        else:
            verts_3d = verts
        
        # build ndc projection matrix
        matrix_ndc = []
        for batch_i in range(calibration_matrix.shape[0]):
            fx, fy = calibration_matrix[batch_i, 0, 0], calibration_matrix[batch_i, 1, 1]
            cx, cy = calibration_matrix[batch_i, 0, 2], calibration_matrix[batch_i, 1, 2]

            matrix_ndc.append(torch.tensor([
                [2*fx/orig_w, 0.0, (orig_w - 2*cx)/orig_w, 0.0],
                [0.0, -2*fy/orig_h, -(orig_h - 2*cy)/orig_h, 0.0],
                [0.0, 0.0, (-far - near) / (far - near), -2.0*far*near/(far-near)],
                [0.0, 0.0, -1.0, 0.0]
            ], device=device))

        matrix_ndc = torch.stack(matrix_ndc, dim=0)
        
        # convert verts to verts ndc
        verts_3d_homo = torch.cat([verts_3d, torch.ones(*verts_3d.shape[:2], 1, device=device)], dim=-1)
        verts_3d_homo[:, :, 2] *= -1  # invert z-axis
        
        verts_ndc = torch.bmm(verts_3d_homo, matrix_ndc.transpose(1, 2))
        
        return verts_ndc, matrix_ndc
    
    def render(self, verts_ndc, matrix_ndc, render_h=256, render_w=256):
        device = verts_ndc.device
        
        rast, rast_db = dr.rasterize(self.glctx, verts_ndc, self.faces, resolution=[render_h, render_w])
        mask = (rast[:, :, :, 2] > 0.0).unsqueeze(-1).type(torch.float32)

        uv, uv_da = dr.interpolate(self.uv_vert_values, rast, self.faces, rast_db=rast_db, diff_attrs='all')

        # invert y-axis
        inv_idx = torch.arange(uv.shape[1] - 1, -1, -1).long().to(device)
        
        uv = uv.index_select(1, inv_idx)
        uv_da = uv_da.index_select(1, inv_idx)
        mask = mask.index_select(1, inv_idx)

        # make channel dim second
        uv = uv.permute(0, 3, 1, 2)
        uv_da = uv_da.permute(0, 3, 1, 2)
        mask = mask.permute(0, 3, 1, 2)
        rast = rast.permute(0, 3, 1, 2)
        
        # norm uv to [-1.0, 1.0]
        uv = 2 * uv - 1
        
        # set empty pixels to -10.0
        uv = uv * mask + (-10.0) * (1 - mask)

        return uv, uv_da, mask, rast
    
    def texture(self, texture, uv, mask=None, uv_da=None, mip=None, filter_mode='auto', boundary_mode='wrap', max_mip_level=None):
        texture = texture.permute(0, 2, 3, 1).contiguous()
        uv = (uv.permute(0, 2, 3, 1).contiguous() + 1) / 2  # norm to [0.0, 1.0]
        
        if uv_da is not None:
            uv_da = uv_da.permute(0, 2, 3, 1).contiguous()

        sampled_texture = dr.texture(
            texture,
            uv,
            uv_da=uv_da,
            mip=mip,
            filter_mode=filter_mode,
            boundary_mode=boundary_mode,
            max_mip_level=max_mip_level,
        )
        
        sampled_texture = sampled_texture.permute(0, 3, 1, 2)
        
        if mask is not None:
            sampled_texture = sampled_texture * mask
        
        return sampled_texture
    
    def antialias(self, color, rast, verts_ndc, topology_hash=None, pos_gradient_boost=1.0):
        color = color.permute(0, 2, 3, 1).contiguous()
        rast = rast.permute(0, 2, 3, 1).contiguous()

        color = dr.antialias(
            color,
            rast,
            verts_ndc,
            self.faces,
            topology_hash=topology_hash,
            pos_gradient_boost=pos_gradient_boost
        )

        color = color.permute(0, 3, 1, 2)

        return color