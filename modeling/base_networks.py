from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from backbones.backbones import load_backbone
from modeling.networks import ConvEncoder, ConvDecoder, Discriminator


class BaseAutoEncoder(ABC, nn.Module):
    def __init__(self, backbone, latent_dim, image_size):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.backbone = backbone

        if self.backbone == "conv":
            self.encoder = ConvEncoder(latent_dim, image_size)
        else:
            self.encoder = load_backbone(backbone, latent_dim, image_size)

        self.decoder = ConvDecoder(latent_dim, image_size)
        self.discriminator = None

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def normalize(self, x):
        return (x - self.mean) / self.std

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def discriminate(self, z):
        if self.training:
            z_noisy = z + 0.1 * torch.randn_like(z)
            return self.discriminator(z_noisy)
        return self.discriminator(z)

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError("Subclasses should implement this method.")


class GenericAAE(BaseAutoEncoder):
    def __init__(self, backbone, latent_dim, image_size):
        super().__init__(backbone, latent_dim, image_size)
        self.discriminator = Discriminator(latent_dim)

    def forward(self, x):
        x = self.normalize(x) if self.backbone != "conv" else x
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


# Adjust encoder in netowrks.py to VAE style
class GenericVAE(BaseAutoEncoder):
    def __init__(self, backbone, latent_dim, image_size):
        super().__init__(backbone, latent_dim, image_size)
        self.discriminator = Discriminator(latent_dim)
        self.noise = 0.015

    def forward(self, x):
        x = self.normalize(x) if self.backbone != "conv" else x
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, z, mu, logvar
