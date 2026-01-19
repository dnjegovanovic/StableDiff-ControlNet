import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from ControlNet.modules.UNet import UNet


class UNetLightning(pl.LightningModule):
    def __init__(self, unet_params, lr=1e-4, weight_decay=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet(UnetParams=unet_params)

    def forward(self, x, t, cond_input=None):
        return self.model(x, t, cond_input=cond_input)

    def training_step(self, batch, batch_idx):
        x = batch["x"]
        t = batch["t"]
        target = batch["target"]
        cond_input = batch.get("cond_input", None)

        pred = self(x, t, cond_input=cond_input)
        loss = F.mse_loss(pred, target)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["x"]
        t = batch["t"]
        target = batch["target"]
        cond_input = batch.get("cond_input", None)

        pred = self(x, t, cond_input=cond_input)
        loss = F.mse_loss(pred, target)

        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
