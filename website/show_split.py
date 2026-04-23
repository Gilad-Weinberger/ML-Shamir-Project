"""
Run from the 'website' folder:
    python show_split.py

Prints exactly which images are used for TRAINING and which for TESTING,
matching the same split the model uses (seed=42, 80/20).
No model is loaded, nothing is retrained.
"""
import os
import torch
from torch.utils.data import random_split
from torchvision import transforms

# ── locate the images folder relative to this script ──────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR  = os.path.join(BASE_DIR, "data", "final_images")

TEST_SPLIT_SEED = 42
TEST_RATIO      = 0.2

# ── minimal inline dataset (no Django needed) ──────────────────────────────
from torch.utils.data import Dataset
from PIL import Image

class _SimpleDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.files  = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        return self.files[idx]   # just the filename

# ── build the split ────────────────────────────────────────────────────────
dataset    = _SimpleDataset(IMAGES_DIR)
n          = len(dataset)
test_size  = max(1, int(n * TEST_RATIO))
train_size = n - test_size

generator = torch.Generator().manual_seed(TEST_SPLIT_SEED)
train_subset, test_subset = random_split(dataset, [train_size, test_size], generator=generator)

train_files = sorted([dataset.files[i] for i in train_subset.indices])
test_files  = sorted([dataset.files[i] for i in test_subset.indices])

# ── print results ──────────────────────────────────────────────────────────
print(f"\nTotal images : {n}")
print(f"Train        : {len(train_files)}  ({100*len(train_files)//n}%)")
print(f"Test         : {len(test_files)}   ({100*len(test_files)//n}%)")

print("\n" + "="*55)
print(f"  TRAIN SET  ({len(train_files)} images)")
print("="*55)
for f in train_files:
    print(f"  {f}")

print("\n" + "="*55)
print(f"  TEST SET   ({len(test_files)} images)")
print("="*55)
for f in test_files:
    print(f"  {f}")

print()
