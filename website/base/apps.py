from django.apps import AppConfig
import torch
import os
import sys

WEBSITE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WEBSITE_ROOT not in sys.path:
    sys.path.insert(0, WEBSITE_ROOT)

from model_config import MODEL_VARIANT, get_images_folder, get_model_file, get_variant_config
from .views import (
    train_grape_leaf_model,
    GrapeLeafRegressor,
)


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
        model_path = os.path.join(os.path.dirname(__file__), get_model_file(MODEL_VARIANT))

        if os.path.exists(model_path):
            print(f"Loading pre-trained model ({MODEL_VARIANT})...")
            model = GrapeLeafRegressor().to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            BaseConfig.model = model
        else:
            print(f"No pre-trained model found for '{MODEL_VARIANT}'. Training new model...")
            BaseConfig.model, _ = train_grape_leaf_model(
                device,
                images_folder=BaseConfig.images_folder,
                model_variant=MODEL_VARIANT,
            )
            torch.save(BaseConfig.model.state_dict(), model_path)
