import os

import torch

from ml.model import GrapeLeafRegressor


def is_hf_configured():
    return bool(os.environ.get("HF_MODEL_REPO", "").strip())


def get_hf_config():
    return {
        "repo": os.environ.get("HF_MODEL_REPO", "").strip(),
        "filename": os.environ.get("HF_MODEL_FILE", "grape_leaf_model_5deg.pth"),
        "token": os.environ.get("HF_TOKEN") or None,
    }


def download_model_from_hub(repo, filename, token=None, cache_dir=None):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo,
        filename=filename,
        token=token,
        cache_dir=cache_dir,
    )


def load_local_model(model_file, device, base_dir=None):
    base_dir = base_dir or os.path.dirname(__file__)
    model_path = os.path.join(base_dir, model_file)

    if not os.path.exists(model_path):
        return None

    model = GrapeLeafRegressor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def load_model(model_file, device, base_dir=None):
    model = load_local_model(model_file, device, base_dir=base_dir)
    if model is not None:
        return model

    if not is_hf_configured():
        return None

    cfg = get_hf_config()
    print(f"Downloading model from Hugging Face Hub ({cfg['repo']})...")
    weights_path = download_model_from_hub(cfg["repo"], cfg["filename"], cfg["token"])
    model = GrapeLeafRegressor().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model
