import os
import tempfile
from abc import ABC
import torch
import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from utils.visualization import plot_visuals, plot_latent_histogram


class BaseLitAutoEncoder(ABC, L.LightningModule):
    def __init__(self, class_name):
        super().__init__()
        self.class_name = class_name

        self.all_x = []
        self.all_recon = []
        self.all_labels = []
        self.all_masks = []
        self.all_error_maps = []

    def _on_fit_start(self):
        if isinstance(self.logger, MLFlowLogger):
            self.run_id = self.logger.run_id
            self.experiment = self.logger.experiment

    def _ensure_logger_setup(self):
        if not hasattr(self, "experiment") or not hasattr(self, "run_id"):
            if isinstance(self.logger, MLFlowLogger):
                self.experiment = self.logger.experiment
                self.run_id = self.logger.run_id
            else:
                print("Logger is not an MLFlowLogger or is not set.")
                return False
        return True

    def _log_losses(self, x, losses, stage):
        for name, value in losses.items():
            self.log(
                f"{stage}_{name}",
                value.detach(),
                batch_size=x.size(0),
                prog_bar=True,
                on_epoch=True,
                on_step=False,
            )

    def _log_stats(self, z_fake, z_real, d_real, d_fake):
        self.log("z_fake_mean", z_fake.mean().detach())
        self.log("z_fake_std", z_fake.std().detach())
        # self.log("z_real_mean", z_real.mean().detach())
        # self.log("z_real_std", z_real.std().detach())
        self.log("d_real_mean", d_real.mean().detach())
        self.log("d_fake_mean", d_fake.mean().detach())

    def _log_latent_histogram(self, z_real, z_fake):
        if not self._ensure_logger_setup():
            return

        with tempfile.TemporaryDirectory() as tmpdir:
            hist_path = os.path.join(tmpdir, f"{self.class_name}_latent_histogram.png")
            plot_latent_histogram(z_real=z_real, z_fake=z_fake, path=hist_path)

            if os.path.exists(hist_path):
                self.experiment.log_artifact(self.run_id, hist_path)

    def _log_visuals(self, data):
        if not self._ensure_logger_setup():
            return

        with tempfile.TemporaryDirectory() as tmpdir:
            orig_path = os.path.join(tmpdir, f"{self.class_name}_originals.png")
            recon_path = os.path.join(tmpdir, f"{self.class_name}_reconstructions.png")
            error_path = os.path.join(tmpdir, f"{self.class_name}_error_maps.png")
            mask_path = os.path.join(tmpdir, f"{self.class_name}_mask_originals.png")
            image_paths = orig_path, recon_path, error_path, mask_path
            plot_visuals(
                image_paths,
                data,
                num_images=8,
            )

            for path in image_paths:
                if os.path.exists(path):
                    self.experiment.log_artifact(self.run_id, path)

    def _calculate_error_map(self, x, masks, labels, recon):
        error_map = torch.abs(x - recon).mean(dim=1, keepdim=True)
        self.all_x.append(x.detach().cpu())
        self.all_recon.append(recon.detach().cpu())
        self.all_labels.extend(labels)
        self.all_error_maps.append(error_map.detach().cpu())
        self.all_masks.append(masks.detach().cpu())

    def _visualize_results(self, precomputed_error=None):
        data = {
            "x": torch.cat(self.all_x, dim=0),
            "recon": torch.cat(self.all_recon, dim=0),
            "labels": self.all_labels,
            "masks": torch.cat(self.all_masks, dim=0),
            "error": (
                precomputed_error
                if precomputed_error is not None
                else torch.cat(self.all_error_maps, dim=0)
            ),
        }
        self._log_visuals(data)
        self.all_x.clear()
        self.all_recon.clear()
        self.all_labels.clear()
        self.all_masks.clear()
        self.all_error_maps.clear()
