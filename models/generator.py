import os
import math
import random

import torch
from torch import nn

from models.styleganv2.modules import (
    PixelNorm, EqualLinear, ConstantInput, StyledConv, ToRGB, StyledConvAInp
)

class Generator(nn.Module):
    def __init__(
            self,
            size=256,
            style_dim=512,
            n_mlp=8,
            channel_multiplier=4,
            output_channels=16,
            blur_kernel=[1, 3, 3, 1],
            lr_mlp=0.01,
            ainp_tensor=None,
            ainp_scales=None,
    ):
        super().__init__()

        if ainp_tensor is None or ainp_scales is None:
            ainp_tensor = None
            ainp_scales = []

        self.size = size
        self.style_dim = style_dim
        self.output_channels = output_channels

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: min(128 * channel_multiplier, 512),
            8: min(128 * channel_multiplier, 512),
            16: min(128 * channel_multiplier, 512),
            32: min(128 * channel_multiplier, 512),
            64: 128 * channel_multiplier,
            128: 64 * channel_multiplier,
            256: 32 * channel_multiplier,
            512: 16 * channel_multiplier,
            1024: 8 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, out_channel=output_channels, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            scale = 2 ** i
            out_channel = self.channels[scale]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            if scale in ainp_scales:
                ainp = torch.nn.functional.interpolate(ainp_tensor, size=(scale, scale), mode='bilinear')
                self.convs.append(
                    StyledConvAInp(
                        out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel, add_input=ainp
                    )
                )
            else:
                self.convs.append(
                    StyledConv(
                        out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                    )
                )

            self.to_rgbs.append(ToRGB(out_channel, style_dim, out_channel=output_channels))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def forward(
            self,
            styles,
            inject_index=None,
            truncation=1,
            truncation_latent=None,
            input_is_latent=False,
            noise=None,
            randomize_noise=True,
    ):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        assert len(styles) == 14


        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)
                ]

        if truncation != 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t


        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]

        elif len(styles) == 2:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)
        elif len(styles) == (len(self.convs) + 2):
            latent = torch.stack(styles, 1)

        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])

        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2

        ntexture = skip

        return ntexture


