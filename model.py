import torch
import pytorch_lightning as pl
from torchmetrics import Accuracy, JaccardIndex, FBetaScore


class SegmentationModel(pl.LightningModule):
    def __init__(self, model, criterion, optimizer, num_classes):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_classes = num_classes

    def setup(self, stage=None):
        # Initialize metrics here to ensure they are on the right device
        device = self.device  # Gets the device model is on
        self.train_metrics = self._init_metrics(device)
        self.val_metrics = self._init_metrics(device)
        self.test_metrics = self._init_metrics(device)

    def forward(self, x):
        return self.model(x)

    def share_step(self, batch, metrics, stage: str, on_step: bool, on_epoch: bool):
        x, y = batch
        y = y.squeeze(1)

        # Forward pass
        y_hat = self.forward(x.to(torch.float32))
        loss = self.criterion(y_hat, y.to(torch.int64))
        prob_mask = torch.softmax(y_hat, dim=1)
        prediction = torch.argmax(prob_mask, dim=1)

        # Compute metrics
        acc_value = metrics["acc"](prediction, y)
        jaccard_index_value = metrics["jaccard_index"](prediction, y)
        fbeta_score_value = metrics["fbeta_score"](prediction, y)

        # Log metrics
        metrics_dict = {
            f"{stage}_loss": loss,
            f"{stage}_acc": acc_value,
            f"{stage}_jaccard": jaccard_index_value,
            f"{stage}_fbeta": fbeta_score_value,
        }
        self.log_dict(
            metrics_dict,
            prog_bar=True,
            sync_dist=True,
            on_step=on_step,
            on_epoch=on_epoch,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self.share_step(
            batch, self.train_metrics, "train", on_step=True, on_epoch=False
        )

    def validation_step(self, batch, batch_idx):
        return self.share_step(
            batch, self.val_metrics, "val", on_step=True, on_epoch=True
        )

    def test_step(self, batch, batch_idx):
        return self.share_step(
            batch, self.test_metrics, "test", on_step=True, on_epoch=True
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_hat = self.forward(x)
        prob_mask = torch.softmax(y_hat, dim=1)
        prediction = torch.argmax(prob_mask, dim=1)
        return prediction

    def configure_optimizers(self):
        return self.optimizer

    def _init_metrics(self, device):
        # Helper function to initialize metrics
        return {
            "acc": Accuracy(
                task="multiclass",
                num_classes=self.num_classes,
                validate_args=True,
                ignore_index=None,
                average="micro",
            ).to(device),
            "jaccard_index": JaccardIndex(
                task="multiclass",
                threshold=0.5,
                num_classes=self.num_classes,
                validate_args=True,
                ignore_index=None,
                average="macro",
            ).to(device),
            "fbeta_score": FBetaScore(
                task="multiclass",
                beta=1.0,
                threshold=0.5,
                num_classes=self.num_classes,
                average="micro",
                ignore_index=None,
                validate_args=True,
            ).to(device),
        }
