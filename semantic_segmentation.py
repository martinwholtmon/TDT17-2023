from pathlib import Path
import timm
import torch.nn as nn
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

def main(): 
    print("Hello World!")

if __name__ == "__main__":
    main()