import pytorch_lightning as pl


class SegmentationModel(pl.LightningModule):
    def __init__(self, model, criterion, optimizer):
        super(SegmentationModel, self).__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.val_losses = []
        self.val_tps = []
        self.val_fps = []
        self.val_fns = []
        self.val_tns = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        images = images.float()
        masks = masks.squeeze(1).long()
        masks_pred = self(images)
        loss = self.criterion(masks_pred, masks)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        images = images.float()
        masks = masks.squeeze(1).long()
        masks_pred = self(images)
        loss = self.criterion(masks_pred, masks)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    #     # convert mask values to probabilities
    #     prob_mask = torch.softmax(masks_pred, dim=1)
    #     pred_labels = torch.argmax(prob_mask, dim=1).long()

    #     # Compute IoU metric
    #     tp, fp, fn, tn = smp.metrics.get_stats(
    #         pred_labels, masks, mode="multiclass", threshold=None
    #     )
    #     iou = smp.metrics.iou_score(
    #         tp, fp, fn, tn, reduction="micro"
    #     )  # 'macro' or 'weighted'
    #     self.log("val_iou", iou, prog_bar=True)

    #     # Save the stats to compute epoch average
    #     self.val_losses.append(loss)
    #     self.val_tps.append(tp)
    #     self.val_fps.append(fp)
    #     self.val_fns.append(fn)
    #     self.val_tns.append(tn)

    #     return {"val_loss": loss, "iou": iou}

    # def on_validation_epoch_end(self):
    #     avg_loss = torch.stack(self.val_losses).mean()
    #     tp_sum = torch.stack(self.val_tps).sum(axis=0)
    #     fp_sum = torch.stack(self.val_fps).sum(axis=0)
    #     fn_sum = torch.stack(self.val_fns).sum(axis=0)
    #     tn_sum = torch.stack(self.val_tns).sum(axis=0)

    #     # Compute IoU for the whole epoch
    #     iou_epoch = smp.metrics.iou_score(
    #         tp_sum, fp_sum, fn_sum, tn_sum, reduction="micro"
    #     )

    #     self.log("val_loss_epoch", avg_loss)
    #     self.log("val_iou_epoch", iou_epoch)

    #     # Clear the lists to free memory
    #     self.val_losses.clear()
    #     self.val_tps.clear()
    #     self.val_fps.clear()
    #     self.val_fns.clear()
    #     self.val_tns.clear()

    def configure_optimizers(self):
        return self.optimizer

    # def train_dataloader(self):
    #     # Return your train dataloader
    #     return self.train_dataloader

    # def val_dataloader(self):
    #     # Return your validation dataloader
    #     return self.val_dataloader
