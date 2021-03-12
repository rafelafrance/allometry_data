"""A pytorch denoising autoencoder."""

from torch import nn


class Autoencoder(nn.Module):
    """Encoder part of the autoencoder."""

    def __init__(self, in_channels=1, out_channels=1, features=32):
        super().__init__()

        self.start = self.double_conv(in_channels, features)

        self.enc1 = self.encoder(features, features * 2)
        self.enc2 = self.encoder(features * 2, features * 4)
        self.enc3 = self.encoder(features * 4, features * 8)
        self.enc4 = self.encoder(features * 8, features * 16)

        self.dec4 = self.decoder(features * 16, features * 8)
        self.dec3 = self.decoder(features * 8, features * 4)
        self.dec2 = self.decoder(features * 4, features * 2)
        self.dec1 = self.decoder(features * 2, features)

        self.end = nn.Conv2d(
            in_channels=features,
            out_channels=out_channels,
            kernel_size=(1, 1))

    def encoder(self, in_channels, out_channels):
        """Build an encoder segment."""
        return nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.double_conv(in_channels, out_channels)
        )

    def decoder(self, in_channels, out_channels):
        """Build a decoder segment"""
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            self.double_conv(in_channels, out_channels),
        )

    @staticmethod
    def double_conv(in_channels, out_channels, mid_channels=None):
        """Create 2 convolutional layers."""
        mid_channels = mid_channels if mid_channels else out_channels

        return nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        """Run the model."""
        x = self.start(x)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.dec4(x)
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)
        x = self.end(x)
        return x
