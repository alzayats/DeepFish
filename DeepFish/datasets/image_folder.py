"""
ImageFolderDataset - A lightweight dataset for inference on arbitrary images.

This dataset scans a folder for image files and returns them in a format
compatible with the DeepFish model wrappers, without requiring ground truth
annotations, CSV files, or mask folders.
"""

import os
import numpy as np
from PIL import Image
import torch


class ImageFolderDataset:
    """Dataset that loads images from a folder for inference.

    Args:
        image_dir: Path to directory containing images
        transform: Torchvision transform to apply to images
        extensions: Tuple of valid image file extensions
    """

    SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

    def __init__(self, image_dir, transform=None, extensions=None):
        self.image_dir = image_dir
        self.transform = transform
        self.extensions = extensions or self.SUPPORTED_EXTENSIONS

        # Scan directory for image files
        self.image_paths = self._scan_for_images()

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir} with extensions {self.extensions}")

        # For compatibility with vis_on_loader
        self.split = "inference"

    def _scan_for_images(self):
        """Scan the image directory for valid image files."""
        image_paths = []

        for filename in sorted(os.listdir(self.image_dir)):
            if filename.lower().endswith(self.extensions):
                image_paths.append(os.path.join(self.image_dir, filename))

        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        filename = os.path.basename(image_path)

        # Load image
        image_pil = Image.open(image_path).convert('RGB')
        image_original = image_pil.copy()

        # Apply transform if provided
        if self.transform is not None:
            image = self.transform(image_pil)
        else:
            image = torch.from_numpy(np.array(image_pil)).permute(2, 0, 1).float() / 255.0

        batch = {
            "images": image,
            "image_original": torch.from_numpy(np.array(image_original)).permute(2, 0, 1).float() / 255.0,
            "meta": {
                "index": index,
                "image_id": index,
                "filename": filename,
                "image_path": image_path,
                "split": self.split
            }
        }

        return batch
