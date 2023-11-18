"""Module to load the cityscapes dataset"""
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets


class CityScapesDatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        # for mask postprocessing
        valid_classes = list(
            filter(lambda x: x.ignore_in_eval is False, self.dataset.classes)
        )
        self.class_names = [x.name for x in valid_classes] + ["void"]
        self.id_map = {
            old_id: new_id
            for (new_id, old_id) in enumerate([x.id for x in valid_classes])
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        for cur_id in label.unique():
            cur_id = cur_id.item()
            if cur_id not in self.id_map.keys():
                label[label == cur_id] = 250
            else:
                label[label == cur_id] = self.id_map[cur_id]
        label[label == 250] = 19
        label = label.squeeze(0)
        return (img, label)


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
