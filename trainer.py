"""Contains the trainer class for training the model."""
import torch
import numpy as np
from tqdm import tqdm, trange


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        scaler=None,
        model_path=None,
        load_model=False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scaler = scaler
        self.model_path = model_path

        if load_model:
            self.load_checkpoint(self.model_path)

    def fit(self, train_loader, val_loader, epochs, save_freq=1):
        start_epoch = self.epoch if hasattr(self, "epoch") else 0
        for epoch in trange(
            start_epoch, epochs, desc=f"Traning Model on {epochs} epochs"
        ):
            self.epoch = epoch
            self._train_one_epoch(train_loader, epoch)
            self._eval_one_epoch(val_loader, epoch)

            # save model
            if self.model_path and epoch % save_freq == 0:
                self.save_checkpoint(self.model_path)

    def _train_one_epoch(self, train_loader, epoch):
        self.model.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            for i, (data, target) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    output = self.model(data)
                    loss = self.criterion(
                        output.float(), target.squeeze(1).long()
                    )  # squeeze(1) to support multi-class segmentation
                self.optimizer.zero_grad()

                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()

                # Update weights
                self.optimizer.step()
                tepoch.set_postfix(loss=loss.item())

    @torch.inference_mode()
    def _eval_one_epoch(self, val_loader, epoch):
        self.model.eval()
        with tqdm(val_loader, unit="val-batch") as tepoch:
            for i, (data, target) in enumerate(tepoch):
                tepoch.set_description("Validation")

                data, target = data.to(self.device), target.to(self.device)
                pred = self.model.predict(data)
                loss = self.criterion(
                    pred.float(), target.squeeze(1).long()
                )  # squeeze(1) to support multi-class segmentation

                # Update tqdm postfix with both loss and mean IoU
                tepoch.set_postfix(loss=loss.item())

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.epoch = checkpoint["epoch"]

    def save_checkpoint(self, filename):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "epoch": self.epoch,
        }
        torch.save(checkpoint, filename)
