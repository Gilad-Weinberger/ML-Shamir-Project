import os
import tempfile
import threading
import time


_client = None
_client_key = None
_client_lock = threading.Lock()


def _log_timing(step, elapsed_ms):
    print(f"[timing] {step}: {elapsed_ms:.1f}ms")


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


def _is_transient_hf_error(exc):
    message = str(exc).lower()
    transient_markers = ("connection", "timeout", "502", "503", "reset", "unavailable")
    return any(marker in message for marker in transient_markers)


def _reset_gradio_client():
    global _client, _client_key
    _client = None
    _client_key = None


def _get_gradio_client(base_url, token):
    global _client, _client_key
    key = (base_url, token or "")
    with _client_lock:
        if _client is None or _client_key != key:
            t0 = time.perf_counter()
            from gradio_client import Client

            _client = Client(base_url, hf_token=token)
            _client_key = key
            _log_timing("gradio_client_init", (time.perf_counter() - t0) * 1000)
        else:
            _log_timing("gradio_client_init", 0.0)
        return _client


def predict_local(model, image_bytes):
    import torch
    import torchvision.transforms as transforms
    from PIL import Image

    from io import BytesIO

    t_total = time.perf_counter()

    t0 = time.perf_counter()
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    _log_timing("local_decode_image", (time.perf_counter() - t0) * 1000)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    t0 = time.perf_counter()
    image_tensor = transform(pil_image).unsqueeze(0)
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    _log_timing("local_preprocess", (time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        predicted_percentage = output.item()
    _log_timing("local_forward_pass", (time.perf_counter() - t0) * 1000)

    result = round(max(0.0, min(100.0, predicted_percentage)), 1)
    _log_timing("predict_local_total", (time.perf_counter() - t_total) * 1000)
    return result


def _parse_remote_result(result):
    if isinstance(result, dict):
        if result.get("error"):
            raise RuntimeError(result["error"])
        if "prediction" in result and result["prediction"] is not None:
            return round(float(result["prediction"]), 1)
    return round(float(result), 1)


def predict_remote(image_bytes, hf_url=None, token=None):
    from gradio_client import handle_file

    t_total = time.perf_counter()

    base_url = (hf_url or os.environ.get("HF_INFERENCE_URL", "")).strip().rstrip("/")
    if not base_url:
        raise ValueError("HF_INFERENCE_URL is not set")

    token = token or get_hf_token()

    suffix = ".jpg"
    tmp_path = None
    result = None
    try:
        t0 = time.perf_counter()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name
        _log_timing("remote_temp_file_write", (time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        client = _get_gradio_client(base_url, token)
        _log_timing("remote_get_client", (time.perf_counter() - t0) * 1000)

        try:
            t0 = time.perf_counter()
            result = client.predict(handle_file(tmp_path), api_name="/predict")
            _log_timing("remote_gradio_predict_call", (time.perf_counter() - t0) * 1000)
        except Exception as exc:
            if _is_transient_hf_error(exc):
                _log_timing("remote_gradio_predict_call", (time.perf_counter() - t0) * 1000)
                _reset_gradio_client()
                t0 = time.perf_counter()
                client = _get_gradio_client(base_url, token)
                _log_timing("remote_get_client_retry", (time.perf_counter() - t0) * 1000)
                t0 = time.perf_counter()
                result = client.predict(handle_file(tmp_path), api_name="/predict")
                _log_timing("remote_gradio_predict_call_retry", (time.perf_counter() - t0) * 1000)
            else:
                raise RuntimeError(_format_hf_error(exc)) from exc
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(_format_hf_error(exc)) from exc
    finally:
        t0 = time.perf_counter()
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        _log_timing("remote_temp_file_cleanup", (time.perf_counter() - t0) * 1000)

    parsed = _parse_remote_result(result)
    _log_timing("predict_remote_total", (time.perf_counter() - t_total) * 1000)
    return parsed


def predict(image_bytes, model=None, hf_url=None, token=None):
    t0 = time.perf_counter()
    if model is not None:
        result = predict_local(model, image_bytes)
        _log_timing("predict_total (local)", (time.perf_counter() - t0) * 1000)
        return result

    if is_remote_inference_configured() or hf_url:
        result = predict_remote(image_bytes, hf_url=hf_url, token=token)
        _log_timing("predict_total (remote)", (time.perf_counter() - t0) * 1000)
        return result

    raise RuntimeError(
        "No model loaded and HF_INFERENCE_URL is not set. "
        "Place a .pth file in website/base/ or configure HF_INFERENCE_URL."
    )
