import torch
import torch.nn as nn
import torch.nn.functional as F


class BackboneWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, latent_dim: int, image_size: int):
        super().__init__()
        self.backbone = backbone
        self.latent_dim = latent_dim

        dummy = torch.randn(1, 3, image_size, image_size)
        with torch.no_grad():
            x = self._forward_features(dummy)
            if isinstance(x, (list, tuple)):
                x = x[0]
            if x.dim() == 4:
                x = F.adaptive_avg_pool2d(x, 1)
            x = torch.flatten(x, 1)
            self.feature_dim = x.shape[1]

        self.fc_mu = nn.Linear(self.feature_dim, latent_dim)
        # self.fc_logvar = nn.Linear(self.feature_dim, latent_dim)  # Uncomment for full VAE

    def _forward_features(self, x):
        backbone = self.backbone

        if hasattr(backbone, "forward_features"):
            return backbone.forward_features(x)

        if hasattr(backbone, "features"):
            return backbone.features(x)

        if isinstance(backbone, nn.Sequential):
            return backbone(x)

        modules = list(backbone.children())
        last_layer = modules[-1] if modules else None

        if isinstance(last_layer, (nn.AdaptiveAvgPool2d, nn.Linear)):
            return nn.Sequential(*modules[:-1])(x)

        return backbone(x)

    def forward(self, x):
        x = self._forward_features(x)
        if isinstance(x, (list, tuple)):
            x = x[0]
        if x.dim() == 4:
            x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        mu = self.fc_mu(x)
        return mu
        # logvar = self.fc_logvar(x) # Uncomment for full VAE
        # return mu, logvar
