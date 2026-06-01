from django.apps import AppConfig
import os
import sys

WEBSITE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WEBSITE_ROOT not in sys.path:
    sys.path.insert(0, WEBSITE_ROOT)

from model_config import MODEL_VARIANT, get_model_file, get_variant_config

from .inference import is_remote_inference_configured


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

        if os.environ.get("VERCEL") == "1":
            BaseConfig.model = None
            print("Vercel deployment: inference via HF_INFERENCE_URL.")
            if is_remote_inference_configured():
                try:
                    from .inference import _get_gradio_client, get_hf_token

                    url = os.environ["HF_INFERENCE_URL"].strip().rstrip("/")
                    _get_gradio_client(url, get_hf_token())
                    print("Pre-warmed HF Gradio client.")
                except Exception as exc:
                    print(f"HF client pre-warm skipped: {exc}")
            return

        try:
            import torch
        except ImportError:
            BaseConfig.model = None
            if is_remote_inference_configured():
                print("PyTorch not installed; will use HF_INFERENCE_URL for inference.")
            else:
                print(
                    "PyTorch not installed. Install torch for local inference "
                    "or set HF_INFERENCE_URL."
                )
            return

        from .model_loader import load_model

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = load_model(model_file, device)

        if model is not None:
            print(f"Loaded local model ({MODEL_VARIANT}).")
            BaseConfig.model = model
        elif is_remote_inference_configured():
            BaseConfig.model = None
            print(
                f"No local model at base/{model_file}; "
                "will fall back to HF_INFERENCE_URL for predictions."
            )
        else:
            BaseConfig.model = None
            print(
                f"No pre-trained model found for '{MODEL_VARIANT}'. "
                f"Place base/{model_file} in website/base/, set HF_MODEL_REPO to download "
                f"from Hugging Face, set HF_INFERENCE_URL for Space inference, or run "
                f"'python train_model.py --variant {MODEL_VARIANT}'."
            )
