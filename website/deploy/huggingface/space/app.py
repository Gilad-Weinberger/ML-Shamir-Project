import os

# Avoid Gradio localhost checks that fail inside HF Space containers.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "0")
os.environ.setdefault("GRADIO_SERVER_NAME", "0.0.0.0")
os.environ.setdefault("GRADIO_SERVER_PORT", os.environ.get("PORT", "7860"))


def _patch_gradio_client_schema():
    """
    HF Spaces can install a gradio_client version whose schema parser crashes on
    boolean JSON-schema nodes (TypeError: argument of type 'bool' is not iterable).
    """
    import gradio_client.utils as gc_utils

    original_get_type = gc_utils.get_type

    def safe_get_type(schema):
        if not isinstance(schema, dict):
            return "Any"
        return original_get_type(schema)

    gc_utils.get_type = safe_get_type

    original_json_schema = gc_utils._json_schema_to_python_type

    def safe_json_schema(schema, defs=None):
        if not isinstance(schema, dict):
            return "Any"
        return original_json_schema(schema, defs)

    gc_utils._json_schema_to_python_type = safe_json_schema


_patch_gradio_client_schema()

import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

from model import GrapeLeafRegressor

HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "")
HF_MODEL_FILE = os.environ.get("HF_MODEL_FILE", "grape_leaf_model_5deg.pth")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def get_model():
    global _model
    if _model is not None:
        return _model

    if not HF_MODEL_REPO:
        raise gr.Error(
            "Missing Space secret HF_MODEL_REPO. "
            "Add it under Space Settings → Repository secrets "
            "(e.g. GiladWeinberger/grape-leaf-5deg), then Factory reboot."
        )

    model = GrapeLeafRegressor().to(device)
    weights_path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=HF_MODEL_FILE,
        token=HF_TOKEN,
    )
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    _model = model
    return _model


def predict(image: Image.Image):
    if image is None:
        raise gr.Error("No image provided")

    model = get_model()

    if image.mode != "RGB":
        image = image.convert("RGB")

    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        value = model(tensor).item()

    return round(max(0.0, min(100.0, value)), 1)


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Leaf image"),
    outputs=gr.Number(label="Downy mildew (%)", precision=1),
    title="Grape Leaf Downy Mildew Assessment",
    description="Upload a leaf image to estimate the affected area percentage (0–100%).",
    api_name="predict",
    flagging_mode="never",
)

print("Pre-loading model at startup...")
get_model()
print("Model ready.")

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", "7860")),
)
