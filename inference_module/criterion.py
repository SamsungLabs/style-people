import os

import torch
import lpips
import kornia

import models


class MAECriterion(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        return torch.mean(torch.abs(pred - target))


class MSECriterion(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        return torch.mean((pred - target) ** 2)
    

class LPIPSCriterion(torch.nn.Module):
    def __init__(self, net='vgg'):
        super().__init__()

        self.lpips = lpips.LPIPS(net=net)
    
    def forward(self, pred, target, valid_mask=None):
        loss_not_reduced = self.lpips(pred.contiguous(), target.contiguous())
        if valid_mask is not None:
            invalid_mask = torch.logical_not(valid_mask)
            loss_not_reduced[invalid_mask] = 0.0

        return loss_not_reduced.mean()


class SSIMCriterion(torch.nn.Module):
    def __init__(self, window_size=5, max_val=1, reduction='mean'):
        super().__init__()

        self.window_size = window_size
        self.max_val = max_val
        self.reduction = reduction

    def forward(self, pred, target):
        loss = kornia.losses.ssim(
            pred,
            target,
            window_size=self.window_size,
            max_val=self.max_val,
            reduction=self.reduction
        ).mean()

        return loss


class UnaryDiscriminatorNonsaturatingCriterion(torch.nn.Module):
    def __init__(self, discriminator, input_shape=(256, 256)):
        super().__init__()

        self.discriminator = discriminator
        self.input_shape = tuple(input_shape)


    def forward(self, pred_image, pred_segm):
        input_dict = {
            'main': {
                'rgb': kornia.resize(pred_image, self.input_shape),
                'segm': kornia.resize(pred_segm, self.input_shape)
            }
        }
        output_dict = self.discriminator(input_dict)

        fake_score = output_dict['d_score']
        loss = models.styleganv2.modules.g_nonsaturating_loss(fake_score)

        return loss


class UnaryDiscriminatorFeatureMatchingCriterion(torch.nn.Module):
    def __init__(
        self,
        discriminator,
        endpoints=['reduction_1', 'reduction_2', 'reduction_3', 'reduction_4', 'reduction_5', 'reduction_6', 'reduction_7', 'reduction_8'],
        input_shape=(256, 256)
    ):
        super().__init__()

        self.discriminator = discriminator
        self.endpoints = endpoints
        self.input_shape = tuple(input_shape) if input_shape else input_shape


    def forward(self, pred_dict, target_dict):
        pred_output_dict = self.discriminator.extract_endpoints({
            'main': {
                'rgb': kornia.resize(pred_dict['rgb'], self.input_shape) if self.input_shape else pred_dict['rgb'],
                'segm': kornia.resize(pred_dict['segm'], self.input_shape) if self.input_shape else pred_dict['segm'],
                'landmarks': pred_dict['landmarks']
            }
        })

        target_output_dict = self.discriminator.extract_endpoints({
            'main': {
                'rgb': kornia.resize(target_dict['rgb'], self.input_shape) if self.input_shape else target_dict['rgb'],
                'segm': kornia.resize(target_dict['segm'], self.input_shape) if self.input_shape else target_dict['segm'],
                'landmarks': target_dict['landmarks']
            }
        })

        loss = 0.0
        for endpoint in self.endpoints:
            loss += torch.abs(pred_output_dict['endpoints'][endpoint] - target_output_dict['endpoints'][endpoint]).mean()

        loss /= len(self.endpoints)

        return {
            'loss': loss,
            'pred_output_dict': pred_output_dict,
            'target_output_dict': target_output_dict
        }