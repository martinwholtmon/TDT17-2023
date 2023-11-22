import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy, FBetaScore, JaccardIndex, MetricCollection


class SegmentationModel(pl.LightningModule):
    def __init__(
        self,
        model,
        num_classes,
        lr=None,
        total_steps=5 * 744,
        ignore_index=0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.num_classes = num_classes
        metrics = MetricCollection(
            [
                Accuracy(
                    task="multiclass",
                    num_classes=self.num_classes,
                    validate_args=True,
                    ignore_index=ignore_index,
                    average="micro",
                ),
                JaccardIndex(
                    task="multiclass",
                    threshold=0.5,
                    num_classes=self.num_classes,
                    validate_args=True,
                    ignore_index=ignore_index,
                    average="micro",
                ),
                FBetaScore(
                    task="multiclass",
                    beta=1.0,
                    threshold=0.5,
                    num_classes=self.num_classes,
                    average="micro",
                    ignore_index=ignore_index,
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

    def _share_step(self, batch, prefix, batch_idx, metric, on_epoch=False):
        """This is custom, but often common to create a function that is shared between training, validation and test step.

        Args:
            batch (Tensor): _description_
            prefix (str): _description_
            batch_idx (int): _description_
            metric (torchmetrics.Metric): Metric to use.
            on_epoch (bool, optional): Logs epoch accumulated metrics. Defaults to False.

        Returns:
            float: loss
        """
        images, targets = batch

        # forward pass and loss
        outputs = self.forward(images)
        loss = self.criterion(outputs, targets)

        # log metrics
        metric_dict = {f"{prefix}_loss": loss}
        metric_dict.update(metric(outputs, targets))
        self.log_dict(metric_dict, prog_bar=True, on_step=True, on_epoch=on_epoch)
        return loss

    def training_step(self, batch, batch_idx):
        return self._share_step(batch, "train", batch_idx, self.train_metrics)

    def validation_step(self, batch, batch_idx):
        return self._share_step(
            batch, "val", batch_idx, self.valid_metrics, on_epoch=True
        )

    def test_step(self, batch, batch_idx):
        return self._share_step(
            batch, "test", batch_idx, self.test_metrics, on_epoch=True
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.hparams.lr, total_steps=int(self.hparams.total_steps)
        )
        return [optimizer], [scheduler]
