import os
import uuid
from datetime import datetime, timezone


def is_supabase_configured():
    return all([
        os.environ.get("SUPABASE_URL"),
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY"),
        os.environ.get("SUPABASE_STORAGE_BUCKET"),
    ])


def get_supabase_client():
    from supabase import create_client

    return create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_ROLE_KEY"],
    )


def upload_image(file_obj, filename=None):
    """
    Upload image bytes to Supabase Storage and return (public_url, object_path).
    """
    bucket = os.environ["SUPABASE_STORAGE_BUCKET"]
    original_name = filename or getattr(file_obj, "name", "upload.jpg")
    ext = os.path.splitext(original_name)[1] or ".jpg"
    date_prefix = datetime.now(timezone.utc).strftime("%Y/%m/%d")
    object_path = f"uploads/{date_prefix}/{uuid.uuid4().hex}{ext}"

    if hasattr(file_obj, "read"):
        data = file_obj.read()
    else:
        data = file_obj

    content_type = getattr(file_obj, "content_type", None) or "image/jpeg"

    try:
        client = get_supabase_client()
        client.storage.from_(bucket).upload(
            object_path,
            data,
            file_options={"content-type": content_type},
        )
    except Exception as exc:
        message = str(exc).lower()
        if "invalid" in message and "api" in message:
            raise RuntimeError(
                "Supabase rejected SUPABASE_SERVICE_ROLE_KEY (Invalid API key). "
                "Use the service_role JWT from Supabase → Project Settings → API — "
                "not an hf_ Hugging Face token. "
                f"Details: {exc}"
            ) from exc
        raise

    public_url = client.storage.from_(bucket).get_public_url(object_path)
    return public_url, object_path
