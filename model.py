import torch
import pytorch_lightning as pl
from torchmetrics import Accuracy, JaccardIndex, FBetaScore


class SegmentationModel(pl.LightningModule):
    def __init__(self, model, criterion, optimizer, num_classes):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.step_outptus = {
            "loss": [],
            "acc": [],
            "jaccard_index": [],
            "fbeta_score": [],
        }
        self.metrics = {
            "acc": Accuracy(
                task="multiclass",
                num_classes=num_classes,
                validate_args=True,
                ignore_index=None,
                average="micro",
            ),
            "jaccard_index": JaccardIndex(
                task="multiclass",
                threshold=0.5,
                num_classes=num_classes,
                validate_args=True,
                ignore_index=None,
                average="macro",
            ),
            "fbeta_score": FBetaScore(
                task="multiclass",
                beta=1.0,
                threshold=0.5,
                num_classes=num_classes,
                average="micro",
                ignore_index=None,
                validate_args=True,
            ),
        }

    def forward(self, x):
        return self.model(x)

    def share_step(self, batch, stage: str):
        x, y = batch
        y = y.squeeze(1)

        # Forward pass
        y_hat = self.forward(x.to(torch.float32))
        loss = self.criterion(y_hat, y.to(torch.int64))
        prob_mask = torch.softmax(y_hat, dim=1)
        prediction = torch.argmax(prob_mask, dim=1)

        # calculate metrics
        acc = self.metrics["acc"](prediction, y)
        jaccard_index = self.metrics["jaccard_index"](prediction, y)
        fbeta_score = self.metrics["fbeta_score"](prediction, y)

        # log metrics
        self.step_outptus["loss"].append(loss)
        self.step_outptus["acc"].append(acc)
        self.step_outptus["jaccard_index"].append(jaccard_index)
        self.step_outptus["fbeta_score"].append(fbeta_score)
        return loss

    def shared_epoch_end(self, stage):
        loss = torch.stack(self.step_outptus["loss"]).mean()
        acc = torch.stack(self.step_outptus["acc"]).mean()
        jaccard_index = torch.stack(self.step_outptus["jaccard_index"]).mean()
        fbeta_score = torch.stack(self.step_outptus["fbeta_score"]).mean()

        # log metrics
        metrics = {
            f"{stage}_loss": loss,
            f"{stage}_accuracy": acc,
            f"{stage}_jaccard_index": jaccard_index,
            f"{stage}_fbeta_score": fbeta_score,
        }
        self.log_dict(metrics, prog_bar=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        return self.share_step(batch, stage="train")

    def on_train_epoch_end(self):
        self.shared_epoch_end("train")

    def validation_step(self, batch, batch_idx):
        return self.share_step(batch, stage="val")

    def on_validation_epoch_end(self):
        self.shared_epoch_end("val")

    def test_step(self, batch, batch_idx):
        return self.share_step(batch, stage="test")

    def on_test_epoch_end(self):
        self.shared_epoch_end("test")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_hat = self.forward(x)
        prob_mask = torch.softmax(y_hat, dim=1)
        prediction = torch.argmax(prob_mask, dim=1)
        return prediction

    def configure_optimizers(self):
        return self.optimizer
