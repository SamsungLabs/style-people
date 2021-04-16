from .base import SegmentationHeadv2
from .base import SegmentationModel
from .decoderv2 import UnetDecoderv2
from .encoders import resnetv2

class Unetv2(SegmentationModel):
    def __init__(
        self,
        in_channels,
        classes,
        activation=None,
        ngf = 64,
        same=False
    ):
        super().__init__()

        self.encoder = resnetv2.ResNetv2Encoder(
            in_channels=in_channels,
            ngf=ngf,
            same=same
        )

        decoder_channels = (256, 128, 64, 32, 16)
        self.decoder = UnetDecoderv2(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=5,
            same=same
        )

        self.segmentation_head = SegmentationHeadv2(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
            same=same
        )

        self.classification_head = None

        # self.name = "u-{}".format(encoder_name)
        self.initialize()

