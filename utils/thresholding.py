import torch
import numpy as np
import cv2


def compute_otsu_threshold(error_maps: list[torch.Tensor]) -> list[torch.Tensor]:
    """
    Applies Otsu's method to compute an optimal global threshold for binarizing the error maps.

    Args:
        error_maps (list[torch.Tensor]): A list of tensors containing pixel-wise anomaly scores
        in the range [0, 1].

    Returns:
        list[torch.Tensor]: A list of binary tensor of the same shape as `error_maps`, with
        values 0 or 1 indicating normal vs. anomalous pixels.
    """
    scaled_errors = (error_maps.view(-1).cpu().numpy() * 255).astype(np.uint8)
    _, binarized = cv2.threshold(
        scaled_errors, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    binarized = torch.tensor(binarized / 255.0, dtype=torch.float32).to(
        error_maps.device
    )
    bin_errors = binarized.view_as(error_maps)
    return bin_errors


def compute_percentile_threshold(
    error_maps: torch.Tensor, percentile: int = 98
) -> torch.Tensor:
    """
    Binarizes the error maps using a global threshold computed from a specified percentile.

    Args:
        error_maps (list[torch.Tensor]): Tensor of shape (B, H, W) or (B, 1, H, W) containing
        pixel-wise anomaly scores.
        percentile (float): Percentile (in the range [0, 100]) used to compute the global threshold.

    Returns:
        list[torch.Tensor]: A list of binary tensors of the same shape as `error_maps`, where
        pixels above the threshold are set to 1.0.
    """
    all_errors = error_maps.view(-1).cpu().numpy()
    threshold = np.percentile(all_errors, percentile)
    bin_errors = (error_maps > threshold).float()
    return bin_errors
