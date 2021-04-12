import torch
import torch.nn as nn
import torch.nn.functional as F

from models.styleganv2.modules import EqualConv2d, EqualConv2dSame
from models.styleganv2.op import FusedLeakyReLU


class Decoderv2Block(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            norm_layer=None,
            same=False
    ):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        ConvLayer = EqualConv2d if not same else EqualConv2dSame

        conv1 = ConvLayer(
            in_channels + skip_channels, 
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        norm1 = norm_layer(out_channels, affine=True)
        relu1 = FusedLeakyReLU(out_channels)
        self.conv1 = nn.Sequential(conv1, norm1, relu1)

        conv2 = ConvLayer(
            out_channels, 
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        norm2 = norm_layer(out_channels, affine=True)
        relu2 = FusedLeakyReLU(out_channels)
        self.conv2 = nn.Sequential(conv2, norm2, relu2)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x



class UnetDecoderv2(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            same=False
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        # combine decoder keyword arguments
        blocks = [
            Decoderv2Block(in_ch, skip_ch, out_ch, same=same)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        features = features[1:]    # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = head
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x
