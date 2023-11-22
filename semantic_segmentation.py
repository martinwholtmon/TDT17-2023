"""Perform semantic segmentation on CityScapes dataset using EfficientViTB3 as backbone"""
import json
import os

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision import transforms

from dataloader import get_dataloaders
from model import SegmentationModel


def custom_target_transform(x):
    return (x * 255).to(torch.long)


def load_config(config_path):
    """Load the configuration from a JSON file."""
    with open(config_path, "r") as file:
        return json.load(file)


def main():
    print("Starting")
    project_dir_path = os.path.dirname(os.path.abspath(__file__))

    # Load params from JSON file
    config = load_config(os.path.join(project_dir_path, "config.json"))

    # Use parameters from the config file
    general_config = config["general"]
    DEV_RUN = general_config.get("DEV_RUN", False)
    IN_CHANNELS = general_config.get("IN_CHANNELS", 3)
    BATCH_SIZE = general_config.get("BATCH_SIZE", 4)
    EPOCHS = general_config.get("EPOCHS", 20)
    LEARNING_RATE = general_config.get("LEARNING_RATE", 1e-5)
    RESOLUTION = general_config.get("RESOLUTION", 1024)
    PIN_MEMORY = general_config.get("PIN_MEMORY", False)
    WORKERS = general_config.get("WORKERS", 0)
    NAME = general_config.get("NAME", "DeepLabV3Plus50")
    CHECKPOINT_NAME = general_config.get("CHECKPOINT_NAME", None)
    CHECKPOINT_DIR = os.path.join(project_dir_path, "checkpoints", NAME)

    # SMP Model parameters
    smp_config = config["smp_model"]
    ENCODER_NAME = smp_config.get("ENCODER_NAME", "resnet50")
    ENCODER_WEIGHTS = smp_config.get("ENCODER_WEIGHTS", "imagenet")
    ACTIVATION = smp_config.get("ACTIVATION", None)

    # Load data
    # Define transformations
    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (RESOLUTION, RESOLUTION)
            ),  # Resize images to a fixed size for training
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Normalize with ImageNet stats
        ]
    )

    target_transforms = transforms.Compose(
        [
            transforms.Resize(
                (RESOLUTION, RESOLUTION),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),  # Resize masks without interpolation
            transforms.ToTensor(),
            custom_target_transform,
        ]
    )
    train_loader, val_loader = get_dataloaders(
        BATCH_SIZE,
        WORKERS,
        PIN_MEMORY,
        train_transform,
        target_transforms,
    )
    CLASSES_TO_PREDICT = len(train_loader.dataset.classes)

    # Create the segmentation model with specified encoder
    smp_model = smp.DeepLabV3Plus(
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=IN_CHANNELS,
        classes=CLASSES_TO_PREDICT,
        activation=ACTIVATION,
    )

    # Initialize your Lightning model
    model = SegmentationModel(
        model=smp_model,
        num_classes=CLASSES_TO_PREDICT,
        lr=LEARNING_RATE,
        total_steps=EPOCHS * len(train_loader),
        ignore_index=0,
    )
    if CHECKPOINT_NAME is not None:
        model = model.load_from_checkpoint(
            os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME), model=smp_model
        )

    # Define callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=CHECKPOINT_DIR,
            filename="{epoch}_{val_loss:.3f}_{val_MulticlassJaccardIndex_micro:.3f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
        ),
        EarlyStopping(
            monitor="val_loss", min_delta=2e-4, patience=8, verbose=False, mode="min"
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    logger = TensorBoardLogger(save_dir="./logs", name=NAME)

    # Initialize a trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        precision=16,
        max_epochs=EPOCHS,
        fast_dev_run=DEV_RUN,
        callbacks=callbacks,
        logger=logger,
        profiler="simple",
    )
    if CHECKPOINT_NAME is None:
        trainer.fit(
            model=model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )
    else:
        trainer.test(model=model, dataloaders=val_loader)


if __name__ == "__main__":
    main()
