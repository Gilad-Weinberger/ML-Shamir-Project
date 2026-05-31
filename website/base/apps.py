from django.apps import AppConfig
import torch
import os
import sys

WEBSITE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WEBSITE_ROOT not in sys.path:
    sys.path.insert(0, WEBSITE_ROOT)

from model_config import MODEL_VARIANT, get_model_file, get_variant_config
from ml.model import GrapeLeafRegressor


class BaseConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'base'
    model = None
    model_variant = MODEL_VARIANT
    images_folder = None

    def ready(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        selected_variant = get_variant_config(MODEL_VARIANT)
        BaseConfig.model_variant = MODEL_VARIANT
        BaseConfig.images_folder = selected_variant["images_folder"]
        model_file = get_model_file(MODEL_VARIANT)
        model_path = os.path.join(os.path.dirname(__file__), model_file)

        if os.path.exists(model_path):
            print(f"Loading pre-trained model ({MODEL_VARIANT})...")
            model = GrapeLeafRegressor().to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            BaseConfig.model = model
        else:
            BaseConfig.model = None
            print(
                f"No pre-trained model found for '{MODEL_VARIANT}'. "
                f"Place base/{model_file} in website/base/, or run "
                f"'python train_model.py' locally or in Google Colab."
            )
