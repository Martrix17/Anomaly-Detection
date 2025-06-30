import random
from collections import defaultdict
import torch
from torchvision.transforms import v2
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def plot_visuals(image_paths: list[str], data: dict, num_images: int = 8):
    x = data["x"].detach().cpu()
    recon = data["recon"].detach().cpu()
    masks = data["masks"].detach().cpu()
    error = data["error"].detach().cpu()
    labels = data["labels"]

    anomaly_to_indices = defaultdict(list)
    for idx, anomaly in enumerate(labels):
        anomaly_to_indices[anomaly].append(idx)

    sampled_indices = _sample_balanced_indices(anomaly_to_indices, num_images)

    _save_image_grids(
        [
            x[sampled_indices],
            recon[sampled_indices],
            error[sampled_indices],
            masks[sampled_indices],
        ],
        image_paths,
    )


def _sample_balanced_indices(anomaly_to_indices, total_samples):
    random.seed(42)
    anomalies = list(anomaly_to_indices.keys())
    samples_per_class = max(1, total_samples // len(anomalies))

    sampled = []
    for anomaly in anomalies:
        indices = anomaly_to_indices[anomaly]
        k = min(samples_per_class, len(indices))
        sampled.extend(random.sample(indices, k))

    remaining = total_samples - len(sampled)
    if remaining > 0:
        all_indices = [
            i
            for indices in anomaly_to_indices.values()
            for i in indices
            if i not in sampled
        ]
        sampled.extend(random.sample(all_indices, min(remaining, len(all_indices))))

    return sampled


def _apply_colormap(gray_image: np.ndarray, cmap: str = "jet") -> torch.Tensor:
    """Convert a grayscale image to a 3-channel heatmap tensor."""
    cm = plt.get_cmap(cmap)
    heatmap = cm(gray_image)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1)
    return heatmap


def _make_pil_grid(images: list[Image.Image], nrow: int = 4) -> Image.Image:
    """Create a PIL grid image from a list of PIL images."""
    widths, heights = zip(*(img.size for img in images))
    max_w, max_h = max(widths), max(heights)
    rows = (len(images) + nrow - 1) // nrow
    grid_img = Image.new("RGB", (nrow * max_w, rows * max_h))

    for idx, img in enumerate(images):
        row = idx // nrow
        col = idx % nrow
        grid_img.paste(img, (col * max_w, row * max_h))

    return grid_img


def _save_image_grids(tensors: list[torch.Tensor], paths: list[str]):
    for tensor, path in zip(tensors, paths):
        if "error" in path:
            images = []
            for i in range(tensor.size(0)):
                error_map = tensor[i].squeeze().numpy()
                heatmap = _apply_colormap(error_map)
                images.append(v2.ToPILImage()(heatmap))

            grid = _make_pil_grid(images, nrow=4)
            grid.save(path)
        else:
            grid = make_grid(tensor, nrow=4, normalize=True)
            v2.ToPILImage()(grid).save(path)


def plot_latent_histogram(z_real: torch.Tensor, z_fake: torch.Tensor, path: str):
    z_real = z_real.detach().cpu().numpy()
    z_fake = z_fake.detach().cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.hist(z_real.flatten(), bins=50, alpha=0.5, label="z_real")
    ax.hist(z_fake.flatten(), bins=50, alpha=0.5, label="z_fake")
    ax.set_title("Latent distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)
