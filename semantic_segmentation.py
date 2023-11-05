"""Perform semantic segmentation on CityScapes dataset using EfficientViTB3 as backbone"""
from pathlib import Path
import timm
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SegmentationModel(nn.Module):
    """The model for semantic segmentation using EfficientViTB3 as backbone"""

    def __init__(self, num_classes) -> None:
        super(SegmentationModel, self).__init__()
        self.backbone = timm.create_model("efficientvit_b3", pretrained=True)
        self.classifier = nn.Conv2d(
            self.backbone.num_features, num_classes, kernel_size=1
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


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
    BATCH_SIZE = 2
    WORKERS = 2
    PIN_MEMORY = True

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
    model = SegmentationModel(num_classes=30).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    history = train(model, train_loader, optimizer, criterion, device, epochs)

    # Save model
    # torch.save(model.state_dict(), "semantic_segmentation.pth")


if __name__ == "__main__":
    main()
