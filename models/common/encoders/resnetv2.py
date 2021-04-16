import torch.nn as nn

from models.styleganv2.modules import EqualConv2d, EqualConv2dSame
from models.styleganv2.op import FusedLeakyReLU
from . import _utils as utils


class EncoderMixin:
    """Add encoder functionality such as:
        - output channels specification of feature tensors (produced by encoder)
        - patching first convolution for arbitrary input channels
    """

    @property
    def out_channels(self):
        """Return channels dimensions for each tensor of forward output of encoder"""
        return self._out_channels[: self._depth + 1]

    def set_in_channels(self, in_channels):
        """Change first convolution chennels"""
        if in_channels == 3:
            return

        self._in_channels = in_channels
        if self._out_channels[0] == 3:
            self._out_channels = tuple([in_channels] + list(self._out_channels)[1:])

        utils.patch_first_conv(model=self, in_channels=in_channels)

    def get_stages(self):
        """Method should be overridden in encoder"""
        raise NotImplementedError

    def make_dilated(self, stage_list, dilation_list):
        stages = self.get_stages()
        for stage_indx, dilation_rate in zip(stage_list, dilation_list):
            utils.replace_strides_with_dilation(
                module=stages[stage_indx],
                dilation_rate=dilation_rate,
            )



class BasicBlockv2(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=False, norm_layer=None, same=False):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        ConvLayer = EqualConv2d if not same else EqualConv2dSame

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = ConvLayer(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes, affine=True)
        self.relu = FusedLeakyReLU(planes)
        self.conv2 = ConvLayer(planes, planes, 3, padding=1, bias=False)
        self.bn2 = norm_layer(planes, affine=True)
        self.stride = stride

        if downsample:
            conv_down = ConvLayer(inplanes, planes, 1, stride=2, bias=False)
            norm_down = norm_layer(planes, affine=True)
            self.downsample = nn.Sequential(conv_down, norm_down)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNetv2Encoder(nn.Module, EncoderMixin):
    def __init__(self, in_channels, ngf=64, norm_layer=None, same=False):
        super().__init__()

        self._out_channels = [in_channels, ngf]
        self._out_channels += [ngf*2**i for i in range(4)]
        self._depth = 5

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        ConvLayer = EqualConv2d if not same else EqualConv2dSame

        self.conv1 = ConvLayer(in_channels, ngf, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = norm_layer(ngf, affine=True)
        self.relu = FusedLeakyReLU(ngf)

        maxpool_layers = []
        if same:
            maxpool_layers.append(nn.ReplicationPad2d(1))
        else:
            maxpool_layers.append(nn.ZeroPad2d(1))
        maxpool_layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        self.maxpool = nn.Sequential(*maxpool_layers)

        block1_1 = BasicBlockv2(ngf, ngf, stride=1, norm_layer=norm_layer, same=same)
        block1_2 = BasicBlockv2(ngf, ngf, stride=1, norm_layer=norm_layer, same=same)
        self.layer1 = nn.Sequential(block1_1, block1_2)

        block2_1 = BasicBlockv2(ngf, ngf*2, stride=2, norm_layer=norm_layer, downsample=True, same=same)
        block2_2 = BasicBlockv2(ngf*2, ngf*2, stride=1, norm_layer=norm_layer, same=same)
        self.layer2 = nn.Sequential(block2_1, block2_2)

        block3_1 = BasicBlockv2(ngf*2, ngf*4, stride=2, norm_layer=norm_layer, downsample=True, same=same)
        block3_2 = BasicBlockv2(ngf*4, ngf*4, stride=1, norm_layer=norm_layer, same=same)
        self.layer3 = nn.Sequential(block3_1, block3_2)

        block4_1 = BasicBlockv2(ngf*4, ngf*8, stride=2, norm_layer=norm_layer, downsample=True, same=same)
        block4_2 = BasicBlockv2(ngf*8, ngf*8, stride=1, norm_layer=norm_layer, same=same)
        self.layer4 = nn.Sequential(block4_1, block4_2)

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.norm1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features


class ResNetv2EncoderCont(nn.Module, EncoderMixin):
    def __init__(self, in_channels, add_channels=1, ngf=64, norm_layer=None, same=False):
        super().__init__()

        self._out_channels = [in_channels, ngf]
        self._out_channels += [ngf*2**i for i in range(4)]
        self._depth = 5

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        ConvLayer = EqualConv2d if not same else EqualConv2dSame

        self.in_channels = in_channels
        self.add_channels = add_channels

        self.conv1 = ConvLayer(in_channels, ngf, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_add = ConvLayer(add_channels, ngf, kernel_size=7, stride=2, padding=3, bias=False)

        self.norm1 = norm_layer(ngf, affine=True)
        self.relu = FusedLeakyReLU(ngf)

        maxpool_layers = []
        if same:
            maxpool_layers.append(nn.ReplicationPad2d(1))
        else:
            maxpool_layers.append(nn.ZeroPad2d(1))
        maxpool_layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        self.maxpool = nn.Sequential(*maxpool_layers)

        block1_1 = BasicBlockv2(ngf, ngf, stride=1, norm_layer=norm_layer, same=same)
        block1_2 = BasicBlockv2(ngf, ngf, stride=1, norm_layer=norm_layer, same=same)
        self.layer1 = nn.Sequential(block1_1, block1_2)

        block2_1 = BasicBlockv2(ngf, ngf*2, stride=2, norm_layer=norm_layer, downsample=True, same=same)
        block2_2 = BasicBlockv2(ngf*2, ngf*2, stride=1, norm_layer=norm_layer, same=same)
        self.layer2 = nn.Sequential(block2_1, block2_2)

        block3_1 = BasicBlockv2(ngf*2, ngf*4, stride=2, norm_layer=norm_layer, downsample=True, same=same)
        block3_2 = BasicBlockv2(ngf*4, ngf*4, stride=1, norm_layer=norm_layer, same=same)
        self.layer3 = nn.Sequential(block3_1, block3_2)

        block4_1 = BasicBlockv2(ngf*4, ngf*8, stride=2, norm_layer=norm_layer, downsample=True, same=same)
        block4_2 = BasicBlockv2(ngf*8, ngf*8, stride=1, norm_layer=norm_layer, same=same)
        self.layer4 = nn.Sequential(block4_1, block4_2)

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.norm1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def forward(self, x, alfa=1.):
        stages = self.get_stages()

        features = []

        x_main = x[:, :self.in_channels]
        x_add = x[:, self.in_channels:]
        assert(x_add.shape[1] == self.add_channels)

        features.append(x_main)
        out_main = self.conv1(x_main)
        out_add = self.conv1_add(x_add)

        out = out_main + alfa*out_add

        out = self.norm1(out)
        out = self.relu(out)
        features.append(out)


        out = self.maxpool(out)
        out = self.layer1(out)
        features.append(out)


        out = self.layer2(out)
        features.append(out)

        out = self.layer3(out)
        features.append(out)

        out = self.layer4(out)
        features.append(out)

        return features