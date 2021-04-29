import torch
from torch import nn

from efficientnet_pytorch import EfficientNet


class EfficientNetLevelEncoder(nn.Module):
    def __init__(
            self,
            output_n_latents=14,
            output_style_dim=512,
            feature2latent_input_size=512,
            model_name='efficientnet-b7',
            pretrained=False
    ):
        """
        Check valid model names here: https://github.com/lukemelas/EfficientNet-PyTorch/blob/761ac94cdbecffca2eecec8cc51ac99afce2025e/efficientnet_pytorch/model.py#L26
        """
        super().__init__()

        self.output_n_latents = output_n_latents
        self.output_style_dim = output_style_dim
        self.feature2latent_input_size = feature2latent_input_size

        self.pretrained = pretrained

        if pretrained:
            self.backbone = EfficientNet.from_pretrained(model_name)
        else:
            self.backbone = EfficientNet.from_name(model_name)

        # feature2latent
        feature_sizes = self._model_name_to_feature_sizes(model_name)

        n_latents_left = output_n_latents
        feature2feature_layers = []
        feature2latent_layers = []
        for i, feature_size in enumerate(feature_sizes):
            if i == len(feature_sizes) - 1:
                output_size = n_latents_left * output_style_dim
                n_latents_left -= n_latents_left
            else:
                output_size = 2 * output_style_dim
                n_latents_left -= 2

            feature2feature_layers.append(nn.Sequential(
                nn.Conv2d(feature_size, feature2latent_input_size, 3, padding=1),
                nn.ReLU(inplace=True)
            ))

            feature2latent_layers.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(feature2latent_input_size, output_size)
            ))

        self.feature2feature_layers = nn.ModuleList(feature2feature_layers)
        self.feature2latent_layers = nn.ModuleList(feature2latent_layers)

        # normalization
        self.register_buffer('imagenet_mean', torch.tensor([0.4914, 0.4822, 0.4465]).view(-1, 1, 1))
        self.register_buffer('imagenet_std', torch.tensor([0.2023, 0.1994, 0.2010]).view(-1, 1, 1))

    @staticmethod
    def _model_name_to_feature_sizes(model_name):
        return {
            'efficientnet-b0': [16, 24, 40, 112, 1280],
            'efficientnet-b1': [16, 24, 40, 112, 1280],
            'efficientnet-b2': [16, 24, 48, 120, 1408],
            'efficientnet-b3': [24, 32, 48, 136, 1536],
            'efficientnet-b4': [24, 32, 56, 160, 1792],
            'efficientnet-b5': [24, 40, 64, 176, 2048],
            'efficientnet-b6': [32, 40, 72, 200, 2304],
            'efficientnet-b7': [32, 48, 80, 224, 2560]
        }[model_name]

    def renormalize_image_imagenet(self, image):
        image = (image + 1.0) / 2  # [-1.0, 1.0] -> [0.0, 1.0]
        image = (image - self.imagenet_mean) / self.imagenet_std
        return image

    def forward(self, x):
        if self.pretrained:
            x = self.renormalize_image_imagenet(x)

        endpoints = self.backbone.extract_endpoints(x)
        endpoint_names = sorted(list(endpoints.keys()))

        cumulative_endpoint_feature = None
        latents = []
        for i, endpoint_name in zip(reversed(range(len(endpoint_names))), endpoint_names[::-1]):
            endpoint_feature = endpoints[endpoint_name]
            feature = self.feature2feature_layers[i](endpoint_feature)

            if cumulative_endpoint_feature is None:
                cumulative_endpoint_feature = feature
            else:
                cumulative_endpoint_feature = feature + nn.functional.upsample(cumulative_endpoint_feature,
                                                                               scale_factor=2.0, mode='bilinear',
                                                                               align_corners=True)

            current_latent = self.feature2latent_layers[i](cumulative_endpoint_feature)

            latents.append(current_latent)

        latent = torch.cat(latents, dim=1)
        latent = latent.view(-1, self.output_n_latents, self.output_style_dim)

        return latent


class EfficientNetEncoder(nn.Module):
    def __init__(
            self,
            output_n_latents=14,
            output_style_dim=512,
            model_name='efficientnet-b7',
            pretrained=False,
            dropout_rate=0.5
    ):
        """
        Check valid model names here: https://github.com/lukemelas/EfficientNet-PyTorch/blob/761ac94cdbecffca2eecec8cc51ac99afce2025e/efficientnet_pytorch/model.py#L26
        """
        super().__init__()

        self.output_n_latents = output_n_latents
        self.output_style_dim = output_style_dim

        self.pretrained = pretrained

        if pretrained:
            self.backbone = EfficientNet.from_pretrained(model_name)
        else:
            self.backbone = EfficientNet.from_name(model_name)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        feature_size = self._model_name_to_feature_size(model_name)
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_size, output_n_latents * output_style_dim)
        )

        # normalization
        self.register_buffer('imagenet_mean', torch.tensor([0.4914, 0.4822, 0.4465]).view(-1, 1, 1))
        self.register_buffer('imagenet_std', torch.tensor([0.2023, 0.1994, 0.2010]).view(-1, 1, 1))

    @staticmethod
    def _model_name_to_feature_size(model_name):
        return {
            'efficientnet-b0': 1280,
            'efficientnet-b1': 1280,
            'efficientnet-b2': 1408,
            'efficientnet-b3': 1536,
            'efficientnet-b4': 1792,
            'efficientnet-b5': 2048,
            'efficientnet-b6': 2304,
            'efficientnet-b7': 2560
        }[model_name]

    def renormalize_image_imagenet(self, image):
        image = (image + 1.0) / 2  # [-1.0, 1.0] -> [0.0, 1.0]
        image = (image - self.imagenet_mean) / self.imagenet_std
        return image

    def forward(self, x):
        if self.pretrained:
            x = self.renormalize_image_imagenet(x)

        x = self.backbone.extract_features(x)
        x = self.avg_pool(x)

        x = x.flatten(start_dim=1)
        x = self.head(x)

        x = x.view(-1, self.output_n_latents, self.output_style_dim)

        return x
