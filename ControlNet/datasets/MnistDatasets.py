from pathlib import Path
from typing import Tuple, List

import torchvision
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch


class MNISTDataset(Dataset):
    """Dataset class for loading MNIST-style image data with proper Path handling.

    Args:
        dataset_split: Partition name ('train'/'test')
        data_root: Root directory containing image folders
        image_extension: File extension for images (default: png)
    """

    def __init__(
        self,
        dataset_split: str,
        data_root: str,
        image_extension: str = "png",
        condition_config=None,
    ):

        self.dataset_split = dataset_split
        self.image_extension = image_extension
        self.image_paths, self.labels = self._load_image_metadata(Path(data_root))

        # Conditioning for the dataset
        self.condition_types = (
            [] if condition_config is None else condition_config["condition_types"]
        )

        print(f"Found {len(self.image_paths)} images for {self.dataset_split} split")

    def _load_image_metadata(self, data_root: Path) -> Tuple[List[Path], List[int]]:
        """Scan directory structure and collect image paths with corresponding labels.

        Directory structure expected:
        {data_root}/{digit_label}/{images}.{image_extension}
        """
        if not data_root.exists():
            raise FileNotFoundError(f"Dataset root directory {data_root} not found")

        image_paths = []
        labels = []

        # Iterate through each digit directory (0-9)
        for digit_dir in tqdm(data_root.iterdir(), desc="Scanning digits"):
            if digit_dir.is_dir():
                label = int(digit_dir.name)

                # Collect all images for current digit
                digit_images = digit_dir.glob(f"*.{self.image_extension}")
                for image_path in digit_images:
                    image_paths.append(image_path)
                    labels.append(label)

        return image_paths, labels

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Load image and convert to normalized tensor."""
        image = Image.open(self.image_paths[index])
        tensor = torchvision.transforms.ToTensor()(image)

        cond_inputs = {}
        if "class" in self.condition_types:
            cond_inputs["class"] = self.labels[index]

        im = 2 * tensor - 1
        if len(self.condition_types) == 0:
            # Normalize to [-1, 1] range
            return im
        else:
            return im, cond_inputs
