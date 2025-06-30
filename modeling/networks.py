import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with instance normalization"""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.activation(x + self.block(x))


class ConvEncoder(nn.Module):
    def __init__(self, latent_dim, image_size):
        super().__init__()
        channels = [3, 16, 32, 64, 128, 256, 512, 1024]

        layers = []
        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]

            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                ResidualBlock(out_ch),
            ]
        self.encoder = nn.Sequential(*layers)

        with torch.no_grad():
            example = torch.randn(1, 3, image_size, image_size)
            output = self.encoder(example)
            self.feature_dim = output.view(1, -1).size(1)

        self.fc_mu = nn.Linear(self.feature_dim, latent_dim)
        # self.fc_logvar = nn.Linear(self.feature_dim, latent_dim)  # Uncomment for full VAE

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        return mu
        # logvar = self.fc_logvar(x)  # Uncomment for full VAE
        # return mu, logvar


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim: int, image_size: int):
        super().__init__()
        self.channels = [1024, 512, 256, 128, 64, 32]
        self.init_size = image_size // (2 ** len(self.channels))

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, self.channels[0] * self.init_size * self.init_size),
            nn.BatchNorm1d(self.channels[0] * self.init_size * self.init_size),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        for i in range(len(self.channels) - 1):
            in_ch = self.channels[i]
            out_ch = self.channels[i + 1]

            layers += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                ResidualBlock(out_ch),
            ]

        self.final_block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(self.channels[-1], 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid(),
        )

        self.decoder = nn.Sequential(*layers, self.final_block)

    def forward(self, z):
        x = self.fc(z).view(-1, self.channels[0], self.init_size, self.init_size)
        return self.decoder(x)


class Discriminator(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, z):
        return self.net(z)
