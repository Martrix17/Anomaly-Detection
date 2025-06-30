import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from torchmetrics.functional.image import structural_similarity_index_measure


class SSIMLoss(nn.Module):
    def __init__(self, data_range=1.0):
        super().__init__()
        self.data_range = data_range

    def forward(self, x, recon):
        ssim_val = structural_similarity_index_measure(recon, x, data_range=1.0)
        return 1 - ssim_val


class MultiScalePerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        vgg = vgg16(weights="DEFAULT").features.eval().float()

        layer_indices = [15, 22]
        self.vgg_layers = nn.ModuleList()

        current_block = nn.Sequential()
        for i, layer in enumerate(vgg):
            current_block.add_module(str(i), layer)
            if i in layer_indices:
                self.vgg_layers.append(current_block)
                current_block = nn.Sequential()

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, recon):
        x = x.float()
        recon = recon.float()
        x_norm = (x - self.mean) / self.std
        recon_norm = (recon - self.mean) / self.std

        loss = 0
        x_features = x_norm
        recon_features = recon_norm

        for i, layer in enumerate(self.vgg_layers):
            x_features = layer(x_features)
            recon_features = layer(recon_features)
            layer_loss = F.mse_loss(x_features, recon_features).float()
            weight = 1.0 / (2**i)
            loss += weight * layer_loss
        return loss


class EdgePreservationLoss(nn.Module):
    """Loss to preserve sharp edges and fine details"""

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "sobel_x",
            torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]).float().unsqueeze(0),
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]).float().unsqueeze(0),
        )

    def forward(self, x, recon):
        x_gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        recon_gray = (
            0.299 * recon[:, 0:1] + 0.587 * recon[:, 1:2] + 0.114 * recon[:, 2:3]
        )

        x_edges_x = F.conv2d(x_gray, self.sobel_x, padding=1)
        x_edges_y = F.conv2d(x_gray, self.sobel_y, padding=1)
        x_edges = torch.sqrt(x_edges_x**2 + x_edges_y**2 + 1e-8)

        recon_edges_x = F.conv2d(recon_gray, self.sobel_x, padding=1)
        recon_edges_y = F.conv2d(recon_gray, self.sobel_y, padding=1)
        recon_edges = torch.sqrt(recon_edges_x**2 + recon_edges_y**2 + 1e-8)

        edge_loss = F.mse_loss(x_edges, recon_edges)

        x_high_freq = x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        recon_high_freq = recon - F.avg_pool2d(
            recon, kernel_size=3, stride=1, padding=1
        )
        detail_loss = F.mse_loss(x_high_freq, recon_high_freq)

        return edge_loss + 0.5 * detail_loss


class ColorConsistencyLoss(nn.Module):
    """Loss to maintain color consistency"""

    def __init__(self):
        super().__init__()

    def forward(self, x, recon):
        x_mean = torch.mean(x, dim=[2, 3], keepdim=True)
        recon_mean = torch.mean(recon, dim=[2, 3], keepdim=True)

        color_loss = F.mse_loss(x_mean, recon_mean)

        x_std = torch.std(x, dim=[2, 3], keepdim=True)
        recon_std = torch.std(recon, dim=[2, 3], keepdim=True)
        saturation_loss = F.mse_loss(x_std, recon_std)

        hist_loss = 0
        for i in range(3):
            x_sorted = torch.sort(x[:, i].flatten(1))[0]
            recon_sorted = torch.sort(recon[:, i].flatten(1))[0]
            hist_loss += F.mse_loss(x_sorted, recon_sorted)

        return color_loss + 0.5 * saturation_loss + 0.3 * hist_loss / 3


class TextureLoss(nn.Module):
    """Texture loss using Gram matrices"""

    def __init__(self):
        super().__init__()

    def gram_matrix(self, x):
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)

    def forward(self, x, recon):
        scales = [1, 0.5]
        loss = 0

        for scale in scales:
            if scale != 1:
                size = (int(x.shape[2] * scale), int(x.shape[3] * scale))
                x_scaled = F.interpolate(
                    x, size=size, mode="bilinear", align_corners=False
                )
                recon_scaled = F.interpolate(
                    recon, size=size, mode="bilinear", align_corners=False
                )
            else:
                x_scaled = x
                recon_scaled = recon

            x_gram = self.gram_matrix(x_scaled)
            recon_gram = self.gram_matrix(recon_scaled)
            loss += F.l1_loss(x_gram, recon_gram) * scale

        return loss


class ReconstructionLoss(nn.Module):
    def __init__(self, recon_weights):
        super().__init__()
        if sum(recon_weights) == 0:
            recon_weights[0] = 1.0
        self.weights = recon_weights

        self.ssim_loss_fn = SSIMLoss() if self.weights[1] > 0 else None
        self.perceptual_loss_fn = (
            MultiScalePerceptualLoss() if self.weights[2] > 0 else None
        )
        self.edge_loss_fn = EdgePreservationLoss() if self.weights[3] > 0 else None
        self.color_loss_fn = ColorConsistencyLoss() if self.weights[4] > 0 else None
        self.texture_loss_fn = TextureLoss() if self.weights[5] > 0 else None

        self.loss_fns = {
            1: self.ssim_loss_fn,
            2: self.perceptual_loss_fn,
            3: self.edge_loss_fn,
            4: self.color_loss_fn,
            5: self.texture_loss_fn,
        }

    def forward(self, x, recon):
        x = torch.clamp(x, 0, 1)
        recon = torch.clamp(recon, 0, 1)

        losses = {}
        if self.weights[0] > 0:
            losses[0] = self.weights[0] * F.mse_loss(recon, x, reduction="mean")

        for idx, loss_fn in self.loss_fns.items():
            if loss_fn is not None:
                losses[idx] = self.weights[idx] * loss_fn(x, recon)

        return sum(losses.values())


def generator_loss(d_fake, smooth_real=0.9):
    target = torch.full_like(d_fake, smooth_real)
    return F.binary_cross_entropy_with_logits(d_fake, target)


def discriminator_loss(d_real, d_fake, smooth_real=0.9):
    real_target = torch.full_like(d_real, smooth_real)
    fake_target = torch.full_like(d_fake, 1 - smooth_real)

    real_loss = F.binary_cross_entropy_with_logits(d_real, real_target)
    fake_loss = F.binary_cross_entropy_with_logits(d_fake, fake_target)
    return 0.5 * (real_loss + fake_loss)
