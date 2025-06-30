import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    jaccard_score,
)


def compute_anomaly_metrics(error_map: torch.Tensor, masks: torch.Tensor) -> dict:
    """
    Computes pixel-level anomaly detection metrics between predicted error maps andground truth
    masks.

    Parameters:
        error_map (torch.Tensor): A tensor representing the predicted anomaly scores per pixel.
        masks (torch.Tensor): A tensor representing the binary ground truth anomaly masks.

    Returns:
        dict: A dictionary containing the following evaluation metrics:
            - "pixel_auroc": Area under the ROC curve.
            - "pixel_auprc": Area under the precision-recall curve.
            - "pixel_f1": F1 score at a 0.5 threshold.
            - "pixel_iou": Intersection over Union (Jaccard index) at a 0.5 threshold.
    """
    preds = error_map.view(error_map.size(0), -1).cpu().numpy()
    gts = masks.view(masks.size(0), -1).cpu().numpy()
    gts = (gts > 0.5).astype(np.uint8)

    return {
        "pixel_auroc": roc_auc_score(gts.ravel(), preds.ravel()),
        "pixel_auprc": average_precision_score(gts.ravel(), preds.ravel()),
        "pixel_f1": f1_score(gts.ravel(), preds.ravel() > 0.5, zero_division=0),
        "pixel_iou": jaccard_score(gts.ravel(), preds.ravel() > 0.5, zero_division=0),
    }
