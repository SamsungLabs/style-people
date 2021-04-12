import torch.nn as nn

from models.styleganv2.modules import EqualConv2d, EqualConv2dSame
from .modules import Flatten, Activation


class SegmentationHeadv2(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1, same=False):
        ConvLayer = EqualConv2d if not same else EqualConv2dSame
        conv2d = ConvLayer(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)
