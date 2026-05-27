import os
import time

import torch
from django.apps import apps
from django.core.management.base import BaseCommand
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from base.views import GrapeLeafDataset, evaluate_model_performance


TEST_SPLIT_SEED = 42
TEST_RATIO = 0.2
EVALUATION_THRESHOLD = 5.0


class Command(BaseCommand):
    help = "Evaluate the grape leaf model on held-out image data and generate evaluation charts."

    def handle(self, *args, **options):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = apps.get_app_config("base").model

        if model is None:
            self.stderr.write(self.style.ERROR("No model is loaded. Run the project once to train or load the model."))
            return

        dataset, source = self._get_evaluation_dataset()
        if dataset is None:
            self.stdout.write("No evaluation data folder (data/test_images, data/final_images or media) found; skipping evaluation.")
            return

        self._run_eval_with_dataloader(model, dataset, device, source)

    def _get_evaluation_dataset(self):
        website_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        test_folder = os.path.join(website_root, "data", "test_images")
        if os.path.isdir(test_folder):
            try:
                test_dataset = GrapeLeafDataset(images_folder=test_folder, transform=transform)
                if len(test_dataset) > 0:
                    return test_dataset, "data/test_images"
            except Exception as e:
                self.stdout.write(f"Evaluation skipped (data/test_images): {e}")

        for folder_name in ("data/final_images", "media"):
            images_folder = os.path.join(website_root, folder_name)
            if not os.path.isdir(images_folder):
                continue

            try:
                full_dataset = GrapeLeafDataset(images_folder=images_folder, transform=transform)
                if len(full_dataset) < 2:
                    continue

                n = len(full_dataset)
                test_size = max(1, int(n * TEST_RATIO))
                train_size = n - test_size
                generator = torch.Generator().manual_seed(TEST_SPLIT_SEED)
                _, test_subset = random_split(full_dataset, [train_size, test_size], generator=generator)
                self.stdout.write(f"\nUsing test set: {len(test_subset)} images (20% of {n} from {folder_name})")
                return test_subset, folder_name
            except Exception as e:
                self.stdout.write(f"Evaluation skipped ({folder_name}): {e}")

        return None, ""

    def _run_eval_with_dataloader(self, model, dataset, device, source=""):
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
        n = len(dataset)
        self.stdout.write(f"\nRunning model evaluation on test set ({n} images{f' from {source}' if source else ''})...")
        t0 = time.perf_counter()
        evaluate_model_performance(model, dataloader, device, threshold=EVALUATION_THRESHOLD)
        elapsed = time.perf_counter() - t0
        self.stdout.write(f"Evaluation completed in {elapsed:.2f} s")
