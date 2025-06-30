import gc
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    CosineAnnealingWarmRestarts,
    CosineAnnealingLR,
)
from modeling.base_networks import GenericAAE
from modeling.base_lit_networks import BaseLitAutoEncoder
from src.loss import generator_loss, discriminator_loss, ReconstructionLoss
from utils.metrics import compute_anomaly_metrics
from utils.thresholding import compute_otsu_threshold, compute_percentile_threshold


class LitAAE(BaseLitAutoEncoder):
    def __init__(
        self,
        class_name,
        learning_rate,
        disc_learning_rate,
        recon_weights,
        backbone,
        latent_dim,
        image_size,
    ):
        super().__init__(class_name)
        self.model = GenericAAE(backbone, latent_dim, image_size)
        self.learning_rate = learning_rate
        self.disc_learning_rate = disc_learning_rate

        self.automatic_optimization = False
        self.opt_enc, self.opt_dec, self.opt_disc = None, None, None
        self.epoch_check = 20

        self.reconstruction_loss = ReconstructionLoss(recon_weights)

    def forward(self, x):
        return self.model(x)

    def reconstruction_step(self, x, recon):
        self.model.encoder.train()
        self.model.decoder.train()
        self.model.discriminator.eval()

        recon_loss = self.reconstruction_loss(x, recon)

        self.opt_enc.zero_grad()
        self.opt_dec.zero_grad()
        self.manual_backward(recon_loss, retain_graph=True)
        clip_grad_norm_(self.model.encoder.parameters(), max_norm=1.0)
        clip_grad_norm_(self.model.decoder.parameters(), max_norm=1.0)
        self.opt_enc.step()
        self.opt_dec.step()

        return recon_loss

    def discriminator_step(self, z_real, z_fake):
        self.model.encoder.eval()
        self.model.decoder.eval()
        self.model.discriminator.train()

        d_real = self.model.discriminate(z_real)
        d_fake = self.model.discriminate(z_fake.detach())
        disc_loss = discriminator_loss(d_real, d_fake)

        self.opt_disc.zero_grad()
        self.manual_backward(disc_loss)
        clip_grad_norm_(self.model.discriminator.parameters(), max_norm=1.0)

        self.opt_disc.step()
        return disc_loss, d_real, d_fake

    def generator_step(self, z_fake):
        self.model.encoder.train()
        self.model.discriminator.eval()

        d_fake = self.model.discriminate(z_fake)
        gen_loss = generator_loss(d_fake)

        self.opt_enc.zero_grad()
        self.manual_backward(gen_loss)
        clip_grad_norm_(self.model.encoder.parameters(), max_norm=1.0)
        self.opt_enc.step()
        return gen_loss

    def training_step(self, batch, batch_idx):
        self.opt_enc, self.opt_dec, self.opt_disc = self.optimizers()
        x, labels, masks = batch

        # Step 1: Reconstruction
        recon, z_fake = self(x)
        recon_loss = self.reconstruction_step(x, recon)

        # Step 2: Discriminator
        z_real = torch.randn_like(z_fake)
        disc_loss, d_real, d_fake = self.discriminator_step(z_real, z_fake)
        d_real_logging = torch.sigmoid(d_real.clone())
        d_fake_logging = torch.sigmoid(d_fake.clone())

        # Step 3: Generator
        z_gen = self.model.encode(x)
        gen_loss = self.generator_step(z_gen)

        self._log_losses(
            x,
            {
                "recon_loss": recon_loss,
                "gen_loss": gen_loss,
                "disc_loss": disc_loss,
            },
            "train",
        )
        self._log_stats(z_fake, z_real, d_real_logging, d_fake_logging)

        if (
            self.current_epoch % self.epoch_check == 0
            and (batch_idx + 1) == self.trainer.num_training_batches
        ):
            self._calculate_error_map(x, masks, labels, recon)
            # self._log_latent_histogram(z_real=z_real, z_fake=z_fake)

    def on_train_epoch_end(self):
        sched_enc = self.lr_schedulers()

        if isinstance(sched_enc, torch.optim.lr_scheduler.ReduceLROnPlateau):
            current_loss = self.trainer.callback_metrics.get("train_recon_loss", None)
            if current_loss is not None:
                sched_enc.step(current_loss)
        elif hasattr(sched_enc, "step"):
            sched_enc.step()

        if self.current_epoch % self.epoch_check == 0:
            self._visualize_results()

        if self.current_epoch % 5 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    def test_predict_step(self, batch, stage):
        x, labels, masks = batch
        recon, _ = self(x)
        recon_loss = self.reconstruction_loss(x, recon)
        self._log_losses(x, {"recon_loss": recon_loss}, stage)
        self._calculate_error_map(x, masks, labels, recon)

    def test_predict_end(self, stage):
        error_maps_raw = torch.cat(self.all_error_maps, dim=0)
        gt_masks = torch.cat(self.all_masks, dim=0)

        # error_maps_bin = compute_otsu_threshold(error_maps_raw)
        # error_maps_bin = compute_percentile_threshold(error_maps_raw, percentile=95)
        # error_maps_bin = compute_percentile_threshold(error_maps_raw, percentile=98)
        error_maps_bin = error_maps_raw

        metrics = compute_anomaly_metrics(error_maps_bin, gt_masks)
        for metric, value in metrics.items():
            self.log(f"{stage}_{metric}", value, prog_bar=True)

        self._visualize_results(precomputed_error=error_maps_bin)

    def test_step(self, batch):
        self.test_predict_step(batch, "test")

    def on_test_epoch_end(self):
        self.test_predict_end("test")

    def predict_step(self, batch):
        self.test_predict_step(batch, "predict")

    def on_predict_epoch_end(self):
        self.test_predict_end("predict")

    def configure_optimizers(self):
        opt_enc = optim.Adam(self.model.encoder.parameters(), lr=self.learning_rate)
        opt_dec = optim.Adam(self.model.decoder.parameters(), lr=self.learning_rate)
        opt_disc = optim.Adam(
            self.model.discriminator.parameters(),
            lr=self.disc_learning_rate,
        )

        sched_enc = {
            "scheduler": ReduceLROnPlateau(
                opt_enc,
                mode="min",
                factor=0.7,
                patience=20,
                min_lr=1e-6,
            ),
            "interval": "epoch",
            "frequency": 1,
        }

        return [opt_enc, opt_dec, opt_disc], [sched_enc]
