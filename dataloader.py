"""Module to load the cityscapes dataset"""
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


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
