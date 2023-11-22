import os
from PIL import Image
import json
import torch
import matplotlib.pyplot as plt
import numpy as np


def load_config(config_path):
    """Load the configuration from a JSON file."""
    with open(config_path, "r") as file:
        return json.load(file)


def load_images(dir_path):
    """Load images from the given directory as RGB PIL Images."""
    images = []
    for filename in os.listdir(dir_path):
        if filename.lower().endswith(
            (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif")
        ):  # Check for image files
            file_path = os.path.join(dir_path, filename)
            with Image.open(file_path) as img:
                rgb_image = img.convert(
                    "RGB"
                )  # Convert to RGB, dropping alpha channel if present
                images.append(rgb_image)
    return images


def custom_target_transform(x):
    return (x * 255).to(torch.long)


def decode_segmap(mask_tensor, color_map) -> np.ndarray:
    """
    Converts a mask (2D tensor of IDs) to an RGB image using the given color map.

    Args:
    - mask_tensor: A 2D tensor where each element is an ID representing a color.
    - color_map: A dictionary mapping IDs to RGB color tuples.

    Returns:
    - A 3D numpy array representing the RGB image.
    """
    height, width = mask_tensor.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)

    # Ensure mask_tensor is on CPU and in numpy format for processing
    if mask_tensor.is_cuda:
        mask_tensor = mask_tensor.cpu()
    mask = mask_tensor.numpy()

    for i in range(height):
        for j in range(width):
            rgb[i, j, :] = color_map[mask[i, j]]
    return rgb


def visualize_dataloader(dataloader):
    # Create iterator for a dataloader
    data_iter = iter(dataloader)

    # Get a batch of data from each loader
    images, labels = next(data_iter)
    labels = labels.unsqueeze(1)  # Add a channel dimension
    batch_size = images.size(0)

    # Set up the plots
    fig, axes = plt.subplots(nrows=2, ncols=batch_size, figsize=(20, 10))
    fig.suptitle("Sample Training Data")

    # Plot training images and masks
    for i in range(batch_size):
        img = images[i].numpy().transpose((1, 2, 0))  # Correcting channel order
        img = (img - img.min()) / (
            img.max() - img.min()
        )  # Normalize to 0-1 if not already

        axes[0, i].imshow(img)
        axes[0, i].axis("off")
        axes[0, i].set_title("Image")
        axes[1, i].imshow(
            decode_segmap(labels[i].squeeze(), dataloader.dataset.color_map)
        )
        axes[1, i].axis("off")
        axes[1, i].set_title("Label")

    # Show the plots
    plt.show()


def visualize_predictions(model, dataloader):
    # Create iterator for the dataloader
    data_iter = iter(dataloader)

    # Get a batch of data from the dataloader
    images, labels = next(data_iter)
    labels = labels.unsqueeze(1)  # Add a channel dimension

    # Make predictions using the model
    model.eval()
    predictions = model(images)

    # Set up the plots
    fig, axes = plt.subplots(nrows=3, ncols=len(images), figsize=(20, 10))
    fig.suptitle("Prediction on a sample from val loader")

    # Plot images, labels, and predictions
    for i, img_tensor in enumerate(images):
        img = img_tensor.numpy().transpose((1, 2, 0))  # Correcting channel order
        img = (img - img.min()) / (
            img.max() - img.min()
        )  # Normalize to 0-1 if not already

        axes[0, i].imshow(img)
        axes[0, i].axis("off")
        axes[0, i].set_title("Image")

        axes[1, i].imshow(
            decode_segmap(labels[i].squeeze(), dataloader.dataset.color_map)
        )
        axes[1, i].axis("off")
        axes[1, i].set_title("Label")

        axes[2, i].imshow(
            decode_segmap(predictions[i].argmax(axis=0), dataloader.dataset.color_map)
        )
        axes[2, i].axis("off")
        axes[2, i].set_title("Prediction")

    # Show the plots
    plt.show()
