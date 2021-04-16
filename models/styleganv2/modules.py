import math
import random

import numpy as np
import torch
from torch import nn, autograd
from torch.nn import functional as F

from .op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualConv2d(nn.Module):
    def __init__(
            self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True, replicate_pad=False,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding
        # self.padval = padval
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

        self.replicate_pad = replicate_pad

        # self.i=0

    def forward(self, input):

        pad = self.padding
        if self.replicate_pad:
            input = F.pad(input, (pad, pad, pad, pad), mode='replicate')
            pad = 0

        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=pad,
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class EqualConv2dSame(nn.Module):
    def __init__(
            self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding
        # self.padval = padval
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

        # self.i=0

    def forward(self, input):
        pad = self.padding
        input = F.pad(input, (pad, pad, pad, pad), mode='replicate')

        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class EqualLinear(nn.Module):
    def __init__(
            self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):

        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


# ============ Normalization

class AdaTexSpade(nn.Module):
    def __init__(self, num_features, segm_tensor, style_dim, kernel_size=1, eps=1e-4):
        super().__init__()
        self.num_features = num_features
        self.weight = self.bias = None
        self.norm_layer = nn.InstanceNorm2d(num_features, eps=eps, affine=False)

        self.segm_tensor = nn.Parameter(segm_tensor, requires_grad=False)
        n_segmchannels = self.segm_tensor.shape[1]
        in_channel = style_dim + n_segmchannels

        self.style_conv = EqualConv2d(
            in_channel,
            num_features,
            kernel_size,
            padding=kernel_size // 2,
        )

    def forward(self, input, style):
        B, C, H, W = input.shape
        out = self.norm_layer(input)

        sB, sC = style.shape
        style = style[..., None, None]
        style = style.expand(sB, sC, H, W)
        segm_tensor = self.segm_tensor.expand(sB, *self.segm_tensor.shape[1:])
        style = torch.cat([style, segm_tensor], dim=1)
        gammas = self.style_conv(style)

        out = out * gammas
        return out


class AdaIn(nn.Module):
    def __init__(self, num_features, style_dim, eps=1e-4):
        super().__init__()
        self.num_features = num_features
        self.weight = self.bias = None
        self.norm_layer = nn.InstanceNorm2d(num_features, eps=eps, affine=False)
        self.modulation = EqualLinear(style_dim, num_features, bias_init=1)

    def forward(self, input, style):
        B, C, H, W = input.shape
        out = self.norm_layer(input)
        gammas = self.modulation(style)
        gammas = gammas[..., None, None]
        out = out * gammas
        return out


class ModulatedSiren2d(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            style_dim,
            demodulate=True,
            is_first=False,
            omega_0=30
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = 1
        self.in_channel = in_channel
        self.out_channel = out_channel

        fan_in = in_channel
        self.scale = 1 / math.sqrt(fan_in)

        if is_first:
            self.demod_scale = 3 * fan_in
        else:
            self.demod_scale = omega_0 ** 2 / 2
        self.omega0 = omega_0

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, 1, 1)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel})'
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        style = self.modulation(style)
        style = style.view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) * self.demod_scale + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, 1, 1
        )

        input = input.view(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, self.out_channel, height, width)

        out = out * self.omega0

        return out


class ModulatedConv2d(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            demodulate=True,
            upsample=False,
            downsample=False,
            blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        style = self.modulation(style)
        style = style.view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class ModulatedConv2dStyleTensor(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            segm_tensor,
            demodulate=True,
            upsample=False,
            downsample=False,
            blur_kernel=[1, 3, 3, 1],
            style_ks=1,
            norm_style=True
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        self.norm_style = norm_style

        self.segm_tensor = nn.Parameter(segm_tensor, requires_grad=False)
        n_segmchannels = self.segm_tensor.shape[1]
        style_in_channel = style_dim + n_segmchannels

        self.style_conv = EqualConv2d(
            style_in_channel,
            in_channel,
            style_ks,
            padding=style_ks // 2,
        )

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        input_mean = input.mean().item()
        input_std = input.std().item()

        sB, sC = style.shape
        style = style[..., None, None]
        style = style.expand(sB, sC, height, width)
        segm_tensor = self.segm_tensor.expand(sB, *self.segm_tensor.shape[1:])
        style = torch.cat([style, segm_tensor], dim=1)

        style = self.style_conv(style)

        style_scale = 1. / math.sqrt(height * width)

        style_mean = style.mean(dim=(2, 3), keepdim=True)
        style_mean = style_mean.view(batch, 1, in_channel, 1, 1)

        weight = self.scale * self.weight
        weight = weight.expand(batch, *weight.shape[1:])
        weight_demod = weight * style_mean

        if self.demodulate:
            demod = torch.rsqrt(weight_demod.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.reshape(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        input = input * style

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)  # * self.scale * style_scale
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)  # * self.scale * style_scale
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)  # * self.scale * style_scale
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        if self.norm_style:
            out = out * self.scale * style_scale

        out_mean = out.mean().item()
        out_std = out.std().item()
        # print()
        # print(input_std, input_mean)
        # print(out_std, out_mean)
        # if input_std != 0:
        #     print('std shift:', out_std / input_std)
        #     if input_mean != 0:
        #         print('std shift norm:', (out_std / input_std) / input_mean)

        return out


# ============ StyledConv

class Sgv1StyledConv(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=False,
            blur_kernel=[1, 3, 3, 1]
    ):
        super().__init__()

        self.conv = EqualConv2d(
            in_channel,
            out_channel,
            kernel_size,
            padding=kernel_size // 2,
        )

        self.norm = AdaIn(in_channel, style_dim)

        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(in_channel)

        if upsample:
            self.up = Upsample(blur_kernel)
        else:
            self.up = None

    def forward(self, input, style, noise=None):
        out = self.noise(input, noise=noise)
        out = self.activate(out)
        out = self.norm(out, style)
        if self.up is not None:
            out = self.up(out)
        out = self.conv(out)
        return out


class AdaTexStyledConv(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            segm_tensor,
            upsample=False,
            blur_kernel=[1, 3, 3, 1],
            add_input=None,
            style_ks=1,
    ):
        super().__init__()

        self.add_input = add_input
        if add_input is not None:
            in_channel = in_channel + add_input.shape[1]
            self.add_input = nn.Parameter(add_input, requires_grad=False)

        self.conv = EqualConv2d(
            in_channel,
            out_channel,
            kernel_size,
            padding=kernel_size // 2,
        )

        self.norm = AdaTexSpade(in_channel, segm_tensor, style_dim, kernel_size=style_ks)

        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(in_channel)

        if upsample:
            self.up = Upsample(blur_kernel)
        else:
            self.up = None

    def forward(self, input, style, noise=None):
        if self.add_input is not None:
            B = input.shape[0]
            ainp = self.add_input.repeat(B, 1, 1, 1)
            input = torch.cat([input, ainp], dim=1)

        out = self.noise(input, noise=noise)
        out = self.activate(out)
        out = self.norm(out, style)
        if self.up is not None:
            out = self.up(out)
        out = self.conv(out)
        return out


class StyledConvStyleTensor(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            segm_tensor,
            upsample=False,
            blur_kernel=[1, 3, 3, 1],
            demodulate=True,
            style_ks=1,
            norm_style=True
    ):
        super().__init__()

        self.conv = ModulatedConv2dStyleTensor(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            segm_tensor,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
            style_ks=style_ks,
            norm_style=norm_style
        )

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out


class StyledConv(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=False,
            blur_kernel=[1, 3, 3, 1],
            demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out


class StyledSiren(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            style_dim,
            demodulate=True,
            is_first=False,
            omega_0=30.
    ):
        super().__init__()

        self.conv = ModulatedSiren2d(
            in_channel,
            out_channel,
            style_dim,
            demodulate=demodulate,
            is_first=is_first,
            omega_0=omega_0
        )

        self.noise = NoiseInjection()

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = torch.sin(out)

        return out


class StyledConvAInp(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=False,
            blur_kernel=[1, 3, 3, 1],
            demodulate=True,
            add_input=None,
            ainp_trainable=False
    ):
        super().__init__()

        self.add_input = add_input
        if add_input is not None:
            in_channel = in_channel + add_input.shape[1]
            self.add_input = nn.Parameter(add_input, requires_grad=ainp_trainable)

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        if self.add_input is not None:
            B = input.shape[0]
            ainp = self.add_input.repeat(B, 1, 1, 1)
            input = torch.cat([input, ainp], dim=1)

        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        out = self.activate(out)

        return out


class SirenResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, style_dim, demodulate=True, is_first=False, omega_0=30.):
        super().__init__()

        self.conv1 = StyledSiren(in_channel, out_channel, style_dim, demodulate=demodulate, is_first=is_first,
                                 omega_0=omega_0)
        self.conv2 = StyledSiren(out_channel, out_channel, style_dim, demodulate=demodulate, omega_0=omega_0)

        self.skip = SineConv1x1(in_channel, out_channel, omega_0=omega_0)

    def forward(self, input, latent):
        out = self.conv1(input, latent)
        out = self.conv2(out, latent)
        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class StyledConv1x1ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, style_dim,
                 demodulate=True):
        super().__init__()

        self.conv1 = StyledConv(in_channel, out_channel, 1, style_dim, demodulate=demodulate)
        self.conv2 = StyledConv(out_channel, out_channel, 1, style_dim, demodulate=demodulate)

        self.skip = ConvLayer(in_channel, out_channel, 1, bias=False)

    def forward(self, input, latent, noise=None):
        if type(noise) == list and len(noise) == 2:
            noise1 = noise[0]
            noise2 = noise[1]
        else:
            noise1 = noise
            noise2 = noise

        if latent.ndim == 3:
            latent1 = latent[:, 0]
            latent2 = latent[:, 1]
        else:
            latent1 = latent
            latent2 = latent

        out = self.conv1(input, latent1, noise=noise1)
        out = self.conv2(out, latent2, noise=noise2)
        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


# ============ ToRGB
class Sgv1ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, out_channel=3, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.norm = AdaIn(in_channel, style_dim)
        self.activate = FusedLeakyReLU(in_channel)
        self.conv = EqualConv2d(in_channel, out_channel, 1)
        self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input, style, skip=None, return_delta=False):
        out = self.activate(input)
        out = self.norm(out, style)
        out = self.conv(out)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)
            delta = out
            out = out + skip

        if return_delta:
            return out, delta
        else:
            return out


class AdaTexToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, segm_tensor, out_channel=3, upsample=True, blur_kernel=[1, 3, 3, 1],
                 style_ks=1):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.norm = AdaTexSpade(in_channel, segm_tensor, style_dim, kernel_size=style_ks)
        self.activate = FusedLeakyReLU(in_channel)
        self.conv = EqualConv2d(in_channel, out_channel, 1)
        self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input, style, skip=None, return_delta=False):
        out = self.activate(input)
        out = self.norm(out, style)
        out = self.conv(out)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)
            delta = out
            out = out + skip

        if return_delta:
            return out, delta
        else:
            return out


class ToRGBStyleTensor(nn.Module):
    def __init__(self,
                 in_channel,
                 style_dim,
                 segm_tensor,
                 out_channel=3,
                 upsample=True,
                 blur_kernel=[1, 3, 3, 1],
                 style_ks=1,
                 norm_style=True):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2dStyleTensor(in_channel, out_channel, 1, style_dim, segm_tensor, demodulate=False,
                                               style_ks=style_ks, norm_style=norm_style)
        self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input, style, skip=None, return_delta=False):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)
            delta = out
            out = out + skip

        if return_delta:
            return out, delta
        else:
            return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, out_channel=3, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, out_channel, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input, style, skip=None, return_delta=False, alfa=1.):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)
            delta = out
            out = alfa * out + skip

        if return_delta:
            return out, delta
        else:
            return out


# ============ Other


class SineConv1x1(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            is_first=False,
            omega_0=30,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_channel = in_channel
        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, 1, 1)
        )

        if is_first:
            a = -1 / self.in_channel
        else:
            a = -np.sqrt(6 / self.in_channel) / self.omega_0
        b = -a

        self.scale = np.abs(b - a) * math.sqrt(1 / 12)

        spectral_tex = torch.load('/Vol1/dbstore/datasets/a.grigorev/gent/smplx_spectral_texture_norm.pth').cuda()
        spectral_tex_inp = spectral_tex[:, 2:100]
        self.spectral_tex_mask = ((spectral_tex ** 2).sum(dim=1) > 0)[0]
        self.sin_input = None

    def forward(self, x):
        weight = self.weight * self.scale
        out = F.conv2d(x, weight)
        mask = self.spectral_tex_mask
        out = torch.sin(self.omega_0 * out)
        return out


class ConvLayer(nn.Sequential):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            downsample=False,
            blur_kernel=[1, 3, 3, 1],
            bias=True,
            activate=True,
            replicate_pad=False
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
                replicate_pad=replicate_pad
            )
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=None, replicate_pad=False, ):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3, replicate_pad=replicate_pad)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True, replicate_pad=replicate_pad)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False, replicate_pad=replicate_pad
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out

# ==================== Old
