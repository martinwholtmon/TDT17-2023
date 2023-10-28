"""Perform semantic segmentation on CityScapes dataset using EfficientViTB3 as backbone"""
from pathlib import Path
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader


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


def get_dataloaders(batch_size: int = 32) -> tuple[DataLoader, DataLoader]:
    """Load the dataset and return the dataloaders

    Args:
        batch_size (int, optional): Size of the batch. Defaults to 32.

    Returns:
        tuple[DataLoader, DataLoader]: train_loader, test_loader
    """
    # root dir
    data_dir = Path(__file__) / "data" / "cityscapes"
    print(f"Loading data from {data_dir}")

    # Load data from directory
    # TODO: Add transforms
    train_dataset = datasets.Cityscapes(
        data_dir, split="train", mode="fine", target_type="semantic"
    )
    test_dataset = datasets.Cityscapes(
        data_dir, split="val", mode="fine", target_type="semantic"
    )

    # Define dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train(model, train_loader, optimizer, criterion, device, epochs) -> list:
    """The training loop

    Args:
        model (nn.Module): PyTorch model
        train_loader (DataLoader): DataLoader for training data
        optimizer (torch.optim): Optimizer
        criterion (torch.nn): Loss function
        device (torch.device): Device to use
        epochs (int): Epochs to train

    Returns:
        list: loss hitory
    """
    history = []
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            history.append(loss.item())  # Save loss for plotting

            # Print info
            if batch_idx % 10 == 0:
                print(
                    f"Epoch: {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}"
                )
    return history


def main():
    # Params
    epochs = 10
    batch_size = 32

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, test_loader = get_dataloaders(batch_size)

    # Define the model
    model = SegmentationModel(num_classes=30).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    history = train(model, train_loader, optimizer, criterion, device, epochs)

    # Save model
    torch.save(model.state_dict(), "semantic_segmentation.pth")


if __name__ == "__main__":
    main()
