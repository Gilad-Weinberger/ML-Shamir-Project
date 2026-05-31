import os
import sys
import time

import torch
from django.apps import apps
from django.core.management.base import BaseCommand
from torch.utils.data import DataLoader
from torchvision import transforms

WEBSITE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if WEBSITE_ROOT not in sys.path:
    sys.path.insert(0, WEBSITE_ROOT)

from model_config import EVALUATION_THRESHOLD, MODEL_VARIANTS, get_eval_folder, get_test_folder
from ml.dataset import GrapeLeafDataset
from base.views import evaluate_model_performance


class Command(BaseCommand):
    help = "Evaluate the grape leaf model on held-out test data and generate evaluation charts."

    def handle(self, *args, **options):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        app_config = apps.get_app_config("base")
        model = app_config.model
        model_variant = getattr(app_config, "model_variant", "original")

        if model is None:
            self.stderr.write(
                self.style.ERROR(
                    "No model is loaded. Train with 'python train_model.py' "
                    "or download weights from Google Colab, then place the .pth in website/base/."
                )
            )
            return

        dataset, source = self._get_evaluation_dataset(model_variant)
        if dataset is None:
            self.stdout.write(f"No test data found for '{model_variant}' model; skipping evaluation.")
            return

        website_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        metrics_folder = get_eval_folder(model_variant)
        metrics_output_dir = os.path.join(website_root, metrics_folder)
        self._run_eval_with_dataloader(model, dataset, device, source, metrics_output_dir)

    def _get_evaluation_dataset(self, model_variant):
        website_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        if model_variant not in MODEL_VARIANTS:
            self.stdout.write(f"Unknown model variant '{model_variant}' for evaluation.")
            return None, ""

        test_folder_name = get_test_folder(model_variant)
        test_folder = os.path.join(website_root, test_folder_name)
        if os.path.isdir(test_folder):
            try:
                test_dataset = GrapeLeafDataset(images_folder=test_folder, transform=transform)
                if len(test_dataset) > 0:
                    return test_dataset, test_folder_name
            except Exception as e:
                self.stdout.write(f"Evaluation skipped ({test_folder_name}): {e}")

        self.stdout.write(
            f"No test images found in {test_folder_name}. "
            f"Run split-train-validation.py to create train/validation/test folders."
        )
        return None, ""

    def _run_eval_with_dataloader(self, model, dataset, device, source="", output_dir="."):
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        n = len(dataset)
        self.stdout.write(
            f"\nRunning model evaluation on test set ({n} images{f' from {source}' if source else ''})..."
        )
        self.stdout.write(f"Device: {device} | Output dir: {output_dir}")
        self.stdout.flush()
        t0 = time.perf_counter()
        evaluate_model_performance(
            model,
            dataloader,
            device,
            threshold=EVALUATION_THRESHOLD,
            output_dir=output_dir,
        )
        elapsed = time.perf_counter() - t0
        self.stdout.write(f"Evaluation completed in {elapsed:.2f} s")
