from django.apps import AppConfig
import torch
import os
import time
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from .views import (
    train_grape_leaf_model,
    GrapeLeafRegressor,
    GrapeLeafDataset,
    evaluate_model_performance,
)

# Fixed seed so the same images are always used as the test set
TEST_SPLIT_SEED = 42
TEST_RATIO = 0.2  # 20% of data used for evaluation


class BaseConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'base'
    model = None  

    def ready(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = os.path.join(os.path.dirname(__file__), "grape_leaf_model.pth")

        if os.path.exists(model_path):
            print("Loading pre-trained model...")
            model = GrapeLeafRegressor().to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            BaseConfig.model = model
        else:
            print("No pre-trained model found. Training new model...")
            BaseConfig.model, _ = train_grape_leaf_model(device)
            torch.save(BaseConfig.model.state_dict(), model_path)  # Save the trained model

        # Run model evaluation at startup (if evaluation data is available)
        self._run_evaluation(device)

    def _run_evaluation(self, device):
        """Run evaluate_model_performance on the test set only (dedicated test folder or 20% split)."""
        website_root = os.path.dirname(os.path.dirname(__file__))
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        # 1. Prefer a dedicated test folder so we only evaluate on held-out data
        test_folder = os.path.join(website_root, "data", "test_images")
        if os.path.isdir(test_folder):
            try:
                test_dataset = GrapeLeafDataset(images_folder=test_folder, transform=transform)
                if len(test_dataset) > 0:
                    self._run_eval_with_dataloader(test_dataset, device, source="data/test_images")
                    return
            except Exception as e:
                print(f"Startup evaluation skipped (data/test_images): {e}")

        # 2. Otherwise use a train/test split from the main data folder (only evaluate on test portion)
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
                print(f"\nUsing test set: {len(test_subset)} images (20% of {n} from {folder_name})")
                self._run_eval_with_dataloader(test_subset, device, source=folder_name)
                return
            except Exception as e:
                print(f"Startup evaluation skipped ({folder_name}): {e}")
        else:
            print("No evaluation data folder (data/test_images, data/final_images or media) found; skipping startup evaluation.")

    def _run_eval_with_dataloader(self, dataset, device, source=""):
        """Helper to run evaluate_model_performance and timing."""
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
        n = len(dataset)
        print(f"\nRunning model evaluation on test set ({n} images{f' from {source}' if source else ''})...")
        t0 = time.perf_counter()
        evaluate_model_performance(BaseConfig.model, dataloader, device, threshold=5.0)
        elapsed = time.perf_counter() - t0
        print(f"Evaluation completed in {elapsed:.2f} s")
