"""Perform semantic segmentation on CityScapes dataset using EfficientViTB3 as backbone"""
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from trainer import Trainer
from unet import UNet


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

    # Params
    LEARNING_RATE = 1e-4
    CLASSES_TO_PREDICT = 19
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 3
    BATCH_SIZE = 2
    WORKERS = 2
    PIN_MEMORY = True
    LOAD_MODEL = False
    MODEL_PATH = (
        Path(__file__).resolve().parent / "checkpoints" / "semantic_segmentation.pth"
    )
    SAVE_FREQ = 1
    print(DEVICE)

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
        BATCH_SIZE, WORKERS, PIN_MEMORY, train_transform, target_transforms
    )

    # Define the model
    model = UNet(in_channels=3, out_channels=CLASSES_TO_PREDICT).to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    # Train the model
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        scaler=scaler,
        model_path=MODEL_PATH,
        load_model=LOAD_MODEL,
    )
    trainer.fit(train_loader, val_loader, epochs=EPOCHS, save_freq=SAVE_FREQ)


if __name__ == "__main__":
    main()
