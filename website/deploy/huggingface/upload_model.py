#!/usr/bin/env python3
"""
Upload the 5deg model weights and metadata to Hugging Face Hub.

Usage (from website/ directory):
  python deploy/huggingface/upload_model.py --repo YOUR_USERNAME/grape-leaf-5deg

Requires: pip install huggingface_hub
Auth:     hf auth login
"""

import argparse
import os
import shutil
import sys

WEBSITE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(WEBSITE_ROOT, "base", "grape_leaf_model_5deg.pth")


def main():
    parser = argparse.ArgumentParser(description="Upload grape leaf model to Hugging Face Hub")
    parser.add_argument(
        "--repo",
        required=True,
        help="Hub repo id, e.g. YOUR_USERNAME/grape-leaf-5deg",
    )
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS,
        help=f"Path to .pth file (default: {DEFAULT_WEIGHTS})",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/update as a private repo",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.weights):
        print(f"Error: weights not found at {args.weights}", file=sys.stderr)
        print("Train first: python train_model.py --variant 5deg", file=sys.stderr)
        sys.exit(1)

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    api = HfApi()
    repo_id = args.repo

    print(f"Creating repo {repo_id} (private={args.private})...")
    api.create_repo(repo_id=repo_id, repo_type="model", private=args.private, exist_ok=True)

    staging = os.path.join(DEPLOY_DIR, ".upload_staging")
    if os.path.isdir(staging):
        shutil.rmtree(staging)
    os.makedirs(staging)

    weights_name = os.path.basename(args.weights)
    shutil.copy2(args.weights, os.path.join(staging, weights_name))
    shutil.copy2(os.path.join(DEPLOY_DIR, "config.json"), os.path.join(staging, "config.json"))

    model_card_src = os.path.join(DEPLOY_DIR, "model_card.md")
    readme_dst = os.path.join(staging, "README.md")
    shutil.copy2(model_card_src, readme_dst)

    print(f"Uploading from {staging}...")
    api.upload_folder(
        folder_path=staging,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload grape_leaf_model_5deg weights and model card",
    )

    shutil.rmtree(staging)
    print(f"Done. Model available at: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
