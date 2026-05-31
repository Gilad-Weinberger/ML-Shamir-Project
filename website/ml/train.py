import copy
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

WEBSITE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WEBSITE_ROOT not in sys.path:
    sys.path.insert(0, WEBSITE_ROOT)

from model_config import (
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_PATIENCE,
    NUM_EPOCHS,
)
from ml.dataset import GrapeLeafDataset
from ml.model import GrapeLeafRegressor


def _get_loader_kwargs(device, batch_size=None):
    if batch_size is None:
        batch_size = 64
    if device.type == "cuda":
        return {"batch_size": batch_size, "num_workers": 2, "pin_memory": True}
    return {"batch_size": batch_size, "num_workers": 0, "pin_memory": False}


def train_grape_leaf_model(
    device,
    images_folder="data/final_images",
    model_variant="original",
    num_epochs=None,
    batch_size=None,
):
    """
    Train the grape leaf regression model.
    Returns:
        tuple: (model, dataloader) where model is the trained model and
               dataloader is the DataLoader for the training dataset.
    """
    if device.type == "cuda":
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Training on CPU")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_folder = os.path.join(images_folder, "train")
    validation_folder = os.path.join(images_folder, "validation")

    if not os.path.isdir(train_folder) or not os.path.isdir(validation_folder):
        raise FileNotFoundError(
            f"Missing training split folders for '{model_variant}'. "
            f"Expected {train_folder} and {validation_folder}. "
            f"Run split-train-validation.py first."
        )

    training_dataset = GrapeLeafDataset(images_folder=train_folder, transform=transform)
    validation_dataset = GrapeLeafDataset(images_folder=validation_folder, transform=transform)

    if len(training_dataset) == 0:
        raise ValueError(f"No training images found in {train_folder}.")

    print(
        f"Training '{model_variant}' model using physically separated folders:\n"
        f"  train: {train_folder} ({len(training_dataset)} images)\n"
        f"  validation: {validation_folder} ({len(validation_dataset)} images)"
    )

    loader_kwargs = _get_loader_kwargs(device, batch_size=batch_size)
    print(f"Batch size: {loader_kwargs['batch_size']}")
    print(f"Max epochs: {num_epochs if num_epochs is not None else NUM_EPOCHS}")
    dataloader = DataLoader(training_dataset, shuffle=True, **loader_kwargs)

    if len(validation_dataset) == 0:
        validation_dataloader = None
        print("No validation images found; training without early stopping.")
    else:
        validation_dataloader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)
        print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE} epochs.")

    model = GrapeLeafRegressor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = NUM_EPOCHS if num_epochs is None else num_epochs
    patience = EARLY_STOPPING_PATIENCE
    min_delta = EARLY_STOPPING_MIN_DELTA
    best_validation_loss = float("inf")
    best_model_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for images, targets in dataloader:
                images = images.to(device)
                targets = targets.to(device).unsqueeze(1)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                pbar.update(1)

        epoch_loss = running_loss / len(dataloader.dataset)

        if validation_dataloader is None:
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}")
            continue

        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for images, targets in validation_dataloader:
                images = images.to(device)
                targets = targets.to(device).unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, targets)
                validation_loss += loss.item() * images.size(0)

        validation_loss = validation_loss / len(validation_dataloader.dataset)
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Validation Loss: {validation_loss:.4f}"
        )

        if validation_loss < best_validation_loss - min_delta:
            best_validation_loss = validation_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch + 1}. "
                f"Best validation loss: {best_validation_loss:.4f}"
            )
            break

    model.load_state_dict(best_model_state)
    return model, dataloader
