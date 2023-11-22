"""Module to load the cityscapes dataset"""
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class CityScapesDatasetWrapper(Dataset):
    def __init__(self, dataset):
        """Initialize the wrapper for the CityScapes dataset."""
        self.dataset = dataset
        self.ignore_train_id = 255
        self.unlabeled_id = 0

        self.classes = [self.dataset.classes[self.unlabeled_id]] + [
            cls for cls in self.dataset.classes if not cls.ignore_in_eval
        ]

        self.class_names = [cls.name for cls in self.classes]
        self.id_map = self._create_id_map()
        self.color_map = self._create_color_map()
        self.mapping_tensor = self._create_mapping_tensor()
        self.reverse_mapping_tensor = self._create_reverse_mapping_tensor()

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Get the image and remapped target for the given index."""
        image, target = self.dataset[idx]
        target = self.mapping_tensor.to(target.device)[target]
        return image, target.squeeze(0)

    def _create_id_map(self):
        """Create a map from original IDs to new IDs."""
        return {
            old_id: new_id
            for new_id, old_id in enumerate(cls.id for cls in self.classes)
        }

    def _create_mapping_tensor(self):
        """Create a tensor for mapping original IDs to new IDs."""
        return create_mapping_tensor(self.id_map, self.unlabeled_id)

    def _create_reverse_mapping_tensor(self):
        """Create a tensor for mapping new IDs back to original IDs."""
        reverse_id_map = {new_id: old_id for old_id, new_id in self.id_map.items()}
        return create_mapping_tensor(reverse_id_map, self.unlabeled_id)

    def _create_color_map(self):
        """Create a color map for visualizing the segmentation masks."""
        color_map = {}
        for cls in self.classes:
            color_map[self.id_map[cls.id]] = cls.color
        return color_map


def create_mapping_tensor(id_map, unlabeled_id=0):
    max_id = max(id_map.keys())
    mapping_tensor = torch.full((max_id + 1,), unlabeled_id, dtype=torch.long)
    for old_id, new_id in id_map.items():
        mapping_tensor[old_id] = new_id
    return mapping_tensor


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
    data_dir = os.path.join(os.path.abspath(""), "data", "cityscapes")
    print(f"Loading data from {data_dir}")

    # Load data from directory
    train_dataset = CityScapesDatasetWrapper(
        datasets.Cityscapes(
            data_dir,
            split="train",
            mode="fine",
            target_type="semantic",
            transform=train_transform,
            target_transform=target_transforms,
        )
    )

    val_dataset = CityScapesDatasetWrapper(
        datasets.Cityscapes(
            data_dir,
            split="val",
            mode="fine",
            target_type="semantic",
            transform=train_transform,
            target_transform=target_transforms,
        )
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
