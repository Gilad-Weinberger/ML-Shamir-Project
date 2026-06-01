import os
import tempfile


def is_remote_inference_configured():
    return bool(os.environ.get("HF_INFERENCE_URL", "").strip())


def get_hf_token():
    """Return HF token only when set; public Spaces should leave HF_TOKEN unset."""
    token = (os.environ.get("HF_TOKEN") or "").strip()
    return token or None


def _format_hf_error(exc):
    message = str(exc).lower()
    if "invalid" in message and ("api" in message or "token" in message or "key" in message):
        return (
            "Hugging Face authentication failed (HF_TOKEN). "
            "If your Space is public, remove HF_TOKEN from Vercel env vars. "
            "If the Space is private, set HF_TOKEN to a Read token from "
            "https://huggingface.co/settings/tokens (not the Supabase key). "
            f"Details: {exc}"
        )
    return f"Prediction failed via Hugging Face Space: {exc}"


def predict_local(model, image_bytes):
    import torch
    import torchvision.transforms as transforms
    from PIL import Image

    from io import BytesIO

    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(pil_image).unsqueeze(0)
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)

    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        predicted_percentage = output.item()

    return round(max(0.0, min(100.0, predicted_percentage)), 1)


def predict_remote(image_bytes, hf_url=None, token=None):
    from gradio_client import Client, handle_file

    base_url = (hf_url or os.environ.get("HF_INFERENCE_URL", "")).strip().rstrip("/")
    if not base_url:
        raise ValueError("HF_INFERENCE_URL is not set")

    token = token or get_hf_token()

    suffix = ".jpg"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        client = Client(base_url, hf_token=token)
        result = client.predict(handle_file(tmp_path), api_name="/predict")
    except Exception as exc:
        raise RuntimeError(_format_hf_error(exc)) from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if isinstance(result, dict):
        if result.get("error"):
            raise RuntimeError(result["error"])
        if "prediction" in result and result["prediction"] is not None:
            return round(float(result["prediction"]), 1)

    return round(float(result), 1)


def predict(image_bytes, model=None, hf_url=None, token=None):
    if model is not None:
        return predict_local(model, image_bytes)

    if is_remote_inference_configured() or hf_url:
        return predict_remote(image_bytes, hf_url=hf_url, token=token)

    raise RuntimeError(
        "No model loaded and HF_INFERENCE_URL is not set. "
        "Place a .pth file in website/base/ or configure HF_INFERENCE_URL."
    )
