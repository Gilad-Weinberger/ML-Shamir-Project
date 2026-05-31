#!/usr/bin/env python3
"""
Standalone training CLI for the grape leaf regression model.
Run from the website/ directory:

    python train_model.py --variant 5deg --device cuda --epochs 60 --batch-size 64
"""
import argparse
import os
import sys

import torch

WEBSITE_ROOT = os.path.dirname(os.path.abspath(__file__))
if WEBSITE_ROOT not in sys.path:
    sys.path.insert(0, WEBSITE_ROOT)

from model_config import MODEL_VARIANT, NUM_EPOCHS, get_images_folder, get_model_file
from ml.train import train_grape_leaf_model


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is available.")
    return torch.device(device_arg)


def train_and_save(variant=None, device="auto", epochs=None, batch_size=None, output=None):
    selected_variant = variant or MODEL_VARIANT
    selected_epochs = NUM_EPOCHS if epochs is None else epochs
    resolved_device = resolve_device(device)
    images_folder = get_images_folder(selected_variant)
    model_file = get_model_file(selected_variant)

    if output:
        model_path = (
            output if os.path.isabs(output) else os.path.join(WEBSITE_ROOT, output)
        )
    else:
        model_path = os.path.join(WEBSITE_ROOT, "base", model_file)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    print(f"Variant: {selected_variant}")
    print(f"Images folder: {images_folder}")
    print(f"Epochs: {selected_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Output: {model_path}")

    model, _ = train_grape_leaf_model(
        resolved_device,
        images_folder=images_folder,
        model_variant=selected_variant,
        num_epochs=selected_epochs,
        batch_size=batch_size,
    )
    torch.save(model.state_dict(), model_path)
    return model_path


def main():
    parser = argparse.ArgumentParser(description="Train the grape leaf regression model.")
    parser.add_argument(
        "--variant",
        default=MODEL_VARIANT,
        help=f"Model variant to train (default: {MODEL_VARIANT} from model_config.py)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to train on (default: auto)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help=f"Maximum training epochs (default: {NUM_EPOCHS} from model_config.py)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size (default: 64)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .pth path relative to website/ or absolute (default: base/<variant model file>)",
    )
    args = parser.parse_args()
    model_path = train_and_save(
        variant=args.variant,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output=args.output,
    )
    print(f"Saved model weights to {model_path}")


if __name__ == "__main__":
    main()
