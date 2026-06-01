import time
from io import BytesIO

from django.apps import apps
from django.http import JsonResponse
from django.views.decorators.http import require_POST

from .forms import ImageUploadForm
from .inference import is_remote_inference_configured, predict
from .supabase_storage import is_supabase_configured, upload_image


def _log_timing(step, elapsed_ms):
    print(f"[timing] {step}: {elapsed_ms:.1f}ms")


def _parse_upload(request):
    t0 = time.perf_counter()
    form = ImageUploadForm(request.POST, request.FILES)
    if not form.is_valid():
        _log_timing("api_form_validation", (time.perf_counter() - t0) * 1000)
        return None, JsonResponse({"error": "Invalid image upload."}, status=400)

    uploaded_file = request.FILES["image"]
    image_bytes = uploaded_file.read()
    _log_timing("api_read_image_bytes", (time.perf_counter() - t0) * 1000)
    return {
        "bytes": image_bytes,
        "name": uploaded_file.name,
        "content_type": uploaded_file.content_type or "image/jpeg",
    }, None


def _model_unavailable_response():
    return JsonResponse(
        {
            "error": (
                "No trained model is loaded. Train with "
                "'python train_model.py --variant 5deg', place the .pth file "
                "in website/base/, set HF_MODEL_REPO to download from Hugging Face, "
                "or set HF_INFERENCE_URL to use a Hugging Face Space."
            )
        },
        status=503,
    )


@require_POST
def predict_api(request):
    """Run inference only — Supabase upload is deferred to upload_api."""
    t_request = time.perf_counter()

    data, error_response = _parse_upload(request)
    if error_response:
        return error_response

    app_config = apps.get_app_config("base")
    model = app_config.model

    if model is None and not is_remote_inference_configured():
        return _model_unavailable_response()

    try:
        t0 = time.perf_counter()
        prediction = predict(data["bytes"], model=model)
        _log_timing("api_predict", (time.perf_counter() - t0) * 1000)
    except Exception as exc:
        return JsonResponse({"error": f"Prediction failed: {exc}"}, status=500)

    _log_timing("api_predict_total", (time.perf_counter() - t_request) * 1000)
    return JsonResponse({"prediction": prediction})


@require_POST
def upload_api(request):
    """Upload image to Supabase Storage — called after prediction is shown."""
    t_request = time.perf_counter()

    if not is_supabase_configured():
        return JsonResponse(
            {
                "error": (
                    "Supabase Storage is not configured. Set SUPABASE_URL, "
                    "SUPABASE_SERVICE_ROLE_KEY, and SUPABASE_STORAGE_BUCKET "
                    "(see website/deploy/vercel/GUIDE.md)."
                )
            },
            status=503,
        )

    data, error_response = _parse_upload(request)
    if error_response:
        return error_response

    buffer = BytesIO(data["bytes"])
    buffer.name = data["name"]
    buffer.content_type = data["content_type"]

    try:
        t0 = time.perf_counter()
        public_url, _ = upload_image(buffer, data["name"])
        _log_timing("api_upload", (time.perf_counter() - t0) * 1000)
    except Exception as exc:
        return JsonResponse(
            {"error": f"Failed to upload image to Supabase: {exc}"},
            status=500,
        )

    _log_timing("api_upload_total", (time.perf_counter() - t_request) * 1000)
    return JsonResponse({"url": public_url, "filename": data["name"]})
