import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class GrapeLeafDataset(Dataset):
    """
    A custom PyTorch Dataset for loading grape leaf images and their corresponding
    white percentage labels. The labels are extracted from the filenames.
    """

    def __init__(self, images_folder="data/final_images", transform=None):
        self.images_folder = images_folder
        self.transform = transform
        self.image_files = [
            f
            for f in os.listdir(images_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.images_folder, image_name)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        try:
            percentage_str = image_name.split("-")[1]
            percentage = (
                float(percentage_str.split(".")[0])
                if "." in percentage_str
                else float(percentage_str)
            )
        except Exception as e:
            print(f"Error parsing percentage from filename {image_name}: {e}")
            percentage = 0.0

        percentage_tensor = torch.tensor(percentage, dtype=torch.float32)
        return image, percentage_tensor
