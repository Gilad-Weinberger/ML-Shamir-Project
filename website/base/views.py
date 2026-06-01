from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

from django.apps import apps
from django.shortcuts import render

from .forms import ImageUploadForm
from .inference import is_remote_inference_configured, predict
from .supabase_storage import is_supabase_configured, upload_image


def Home(request):
    """
    Handle image upload: run inference and Supabase storage in parallel when possible.
    """
    prediction = None
    uploaded_image_url = None
    uploaded_image_name = None
    model_error = None
    upload_error = None

    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES["image"]
            image_bytes = uploaded_file.read()
            uploaded_image_name = uploaded_file.name

            app_config = apps.get_app_config("base")
            model = app_config.model

            if model is None and not is_remote_inference_configured():
                model_error = (
                    "No trained model is loaded. Train with "
                    "'python train_model.py --variant 5deg', place the .pth file "
                    "in website/base/, set HF_MODEL_REPO to download from Hugging Face, "
                    "or set HF_INFERENCE_URL to use a Hugging Face Space."
                )
            else:
                upload_future = None
                with ThreadPoolExecutor(max_workers=2) as executor:
                    predict_future = executor.submit(predict, image_bytes, model=model)

                    if is_supabase_configured():
                        buffer = BytesIO(image_bytes)
                        buffer.name = uploaded_file.name
                        buffer.content_type = uploaded_file.content_type or "image/jpeg"
                        upload_future = executor.submit(upload_image, buffer, uploaded_file.name)

                    try:
                        prediction = predict_future.result()
                    except Exception as exc:
                        model_error = f"Prediction failed: {exc}"

                    if model_error is None and prediction is not None:
                        if not is_supabase_configured():
                            upload_error = (
                                "Supabase Storage is not configured. Set SUPABASE_URL, "
                                "SUPABASE_SERVICE_ROLE_KEY, and SUPABASE_STORAGE_BUCKET "
                                "(see website/deploy/vercel/GUIDE.md)."
                            )
                        elif upload_future is not None:
                            try:
                                uploaded_image_url, _ = upload_future.result()
                            except Exception as exc:
                                upload_error = f"Failed to upload image to Supabase: {exc}"
    else:
        form = ImageUploadForm()

    context = {
        "form": form,
        "prediction": prediction,
        "uploaded_image_url": uploaded_image_url,
        "uploaded_image_name": uploaded_image_name,
        "model_error": model_error,
        "upload_error": upload_error,
    }
    return render(request, "base/home.html", context)
