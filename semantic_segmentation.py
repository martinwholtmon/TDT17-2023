"""Perform semantic segmentation on CityScapes dataset using EfficientViTB3 as backbone"""
from pathlib import Path

import torch
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SegmentationModel


def get_dataloaders(
    batch_size: int = 32,
    workers: int = 2,
    pin_memory: bool = True,
    train_transform: transforms.Compose = None,
    target_transforms: transforms.Compose = None,
) -> tuple[DataLoader, DataLoader]:
    """Load the dataset from file

    Args:
        batch_size (int, optional): Size of the batch. Defaults to 32.

    Returns:
        tuple[DataLoader, DataLoader]: train_loader, val_loader
    """
    # root dir
    data_dir = Path(__file__).parent / "data" / "cityscapes"
    print(f"Loading data from {data_dir}")

    # Load data from directory
    train_dataset = datasets.Cityscapes(
        data_dir,
        split="train",
        mode="fine",
        target_type="semantic",
        transform=train_transform,
        target_transform=target_transforms,
    )
    val_dataset = datasets.Cityscapes(
        data_dir,
        split="val",
        mode="fine",
        target_type="semantic",
        transform=train_transform,
        target_transform=target_transforms,
    )

    # Define dataloaders
    print(f"Loaded. Creating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=workers,
    )

    # Print info about loaded data
    print(f"Train examples: {len(train_loader.dataset)}")
    print(f"Val examples: {len(val_loader.dataset)}")
    return train_loader, val_loader


def main():
    print("Starting")
    torch.set_float32_matmul_precision("medium")  # 'medium' | 'high'

    # Params
    IN_CHANNELS = 3
    LEARNING_RATE = 1e-4
    CLASSES_TO_PREDICT = 35
    EPOCHS = 10
    BATCH_SIZE = 4
    WORKERS = 4
    PIN_MEMORY = True
    LOAD_MODEL = False
    MODEL_PATH = (
        Path(__file__).resolve().parent
        / "checkpoints"
        / "semantic_segmentation_main.pth"
    )

    # Load data
    # Define transformations
    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),  # Resize images to a fixed size for training
            transforms.RandomRotation(35),
            transforms.RandomHorizontalFlip(0.5),  # Randomly flip images
            transforms.RandomVerticalFlip(0.5),  # Randomly flip images
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Normalize with ImageNet stats
        ]
    )

    target_transforms = transforms.Compose(
        [
            transforms.Resize(
                (256, 256), interpolation=transforms.InterpolationMode.NEAREST
            ),  # Resize masks without interpolation
            transforms.ToTensor(),
        ]
    )
    train_loader, val_loader = get_dataloaders(
        BATCH_SIZE,
        WORKERS,
        PIN_MEMORY,
        train_transform,
        target_transforms,
    )

    # Define the model, for example, a DeepLabV3 with a ResNet-101 encoder
    ENCODER_NAME = "resnet101"
    ENCODER_WEIGHTS = "imagenet"  # No pre-trained weights
    ACTIVATION = (
        None  # Could be None for logits or 'softmax2d' for multiclass segmentation
    )

    # Create the segmentation model with specified encoder
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=IN_CHANNELS,
        classes=CLASSES_TO_PREDICT,
        activation=ACTIVATION,
    )
    # preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER_NAME, ENCODER_WEIGHTS)

    # Define the loss and optimizer function
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Initialize your Lightning model
    model = SegmentationModel(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        num_classes=CLASSES_TO_PREDICT,
    )

    # Define callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=f"checkpoints/DeepLabV3Plus",
            filename="{epoch}_{val_loss:.2f}_{val_accuracy:.2f}",
            save_top_k=10,
            monitor="val_loss",
            mode="min",
        ),
        EarlyStopping(
            monitor="val_loss", min_delta=2e-4, patience=8, verbose=False, mode="min"
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Initialize a trainer
    trainer = pl.Trainer(max_epochs=EPOCHS, callbacks=callbacks)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
