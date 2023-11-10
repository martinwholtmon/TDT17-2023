import torch
import pytorch_lightning as pl
from torchmetrics import MetricCollection, Accuracy, JaccardIndex, FBetaScore


class SegmentationModel(pl.LightningModule):
    def __init__(self, model, criterion, optimizer, num_classes):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_classes = num_classes
        metrics = MetricCollection(
            [
                Accuracy(
                    task="multiclass",
                    num_classes=self.num_classes,
                    validate_args=True,
                    ignore_index=None,
                    average="micro",
                ),
                JaccardIndex(
                    task="multiclass",
                    threshold=0.5,
                    num_classes=self.num_classes,
                    validate_args=True,
                    ignore_index=None,
                    average="macro",
                ),
                FBetaScore(
                    task="multiclass",
                    beta=1.0,
                    threshold=0.5,
                    num_classes=self.num_classes,
                    average="micro",
                    ignore_index=None,
                    validate_args=True,
                ),
            ],
            postfix="_micro",
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(self, x):
        return self.model(x)

    def share_step(self, batch, stage: str, on_step: bool, on_epoch: bool):
        x, y = batch

        # Forward pass
        y_hat = self.forward(x.to(torch.float32))
        loss = self.criterion(y_hat, y.to(torch.int64))
        prob_mask = torch.softmax(y_hat, dim=1)
        prediction = torch.argmax(prob_mask, dim=1)

        # Log metrics
        match stage:
            case "train":
                output = self.train_metrics(prediction, y)
            case "val":
                output = self.valid_metrics(prediction, y)
            case "test":
                output = self.test_metrics(prediction, y)
            case _:
                raise ValueError(f"Invalid stage: {stage}")

        # Add loss to the output dictionary
        output[f"{stage}_loss"] = loss

        self.log_dict(
            output,
            prog_bar=True,
            sync_dist=True,
            on_step=on_step,
            on_epoch=on_epoch,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self.share_step(batch, "train", on_step=True, on_epoch=False)

    def validation_step(self, batch, batch_idx):
        return self.share_step(batch, "val", on_step=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        return self.share_step(batch, "test", on_step=True, on_epoch=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_hat = self.forward(x)
        prob_mask = torch.softmax(y_hat, dim=1)
        prediction = torch.argmax(prob_mask, dim=1)
        return prediction

    def configure_optimizers(self):
        return self.optimizer
