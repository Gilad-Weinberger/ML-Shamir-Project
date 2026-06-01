from django.apps import AppConfig
import os
import sys

WEBSITE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WEBSITE_ROOT not in sys.path:
    sys.path.insert(0, WEBSITE_ROOT)

from model_config import MODEL_VARIANT, get_model_file, get_variant_config


class BaseConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'base'
    model = None
    model_variant = MODEL_VARIANT
    images_folder = None

    def ready(self):
        selected_variant = get_variant_config(MODEL_VARIANT)
        BaseConfig.model_variant = MODEL_VARIANT
        BaseConfig.images_folder = selected_variant["images_folder"]
        model_file = get_model_file(MODEL_VARIANT)

        if os.environ.get("HF_INFERENCE_URL", "").strip():
            BaseConfig.model = None
            print("Remote inference enabled (HF_INFERENCE_URL); skipping local model load.")
            return

        try:
            import torch
        except ImportError:
            BaseConfig.model = None
            print(
                "PyTorch not installed. Set HF_INFERENCE_URL for remote inference, "
                "or install torch for local inference."
            )
            return

        from .model_loader import load_model

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = load_model(model_file, device)

        if model is not None:
            print(f"Loaded pre-trained model ({MODEL_VARIANT}).")
            BaseConfig.model = model
        else:
            BaseConfig.model = None
            print(
                f"No pre-trained model found for '{MODEL_VARIANT}'. "
                f"Place base/{model_file} in website/base/, set HF_MODEL_REPO to download "
                f"from Hugging Face, set HF_INFERENCE_URL for Space inference, or run "
                f"'python train_model.py --variant {MODEL_VARIANT}'."
            )
