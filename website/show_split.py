"""
Run from the 'website' folder:
    python show_split.py

Prints which images are in:
- data/final_images/train
- data/final_images/validation
- data/final_images/test

Evaluation charts are saved to evals/<variant>/ folders, not input images.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from model_config import MODEL_VARIANT, get_eval_folder, get_images_folder, get_test_folder


def list_images(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
        and os.path.isfile(os.path.join(folder, f))
    )


images_dir = get_images_folder(MODEL_VARIANT)
train_dir = os.path.join(images_dir, "train")
validation_dir = os.path.join(images_dir, "validation")
test_dir = get_test_folder(MODEL_VARIANT)
metrics_dir = get_eval_folder(MODEL_VARIANT)

train_files = list_images(train_dir)
validation_files = list_images(validation_dir)
test_files = list_images(test_dir)

print(f"\nUsing MODEL_VARIANT='{MODEL_VARIANT}'")
print(f"Dataset folder: {images_dir}")
print(f"Train folder  : {len(train_files)}")
print(f"Validation    : {len(validation_files)}")
print(f"Test folder   : {len(test_files)}")
print(f"Metrics output: {metrics_dir}/")

print("\n" + "=" * 55)
print(f"  TRAIN FOLDER  ({len(train_files)} images)")
print("=" * 55)
for filename in train_files:
    print(f"  {filename}")

print("\n" + "=" * 55)
print(f"  VALIDATION FOLDER  ({len(validation_files)} images)")
print("=" * 55)
for filename in validation_files:
    print(f"  {filename}")

print("\n" + "=" * 55)
print(f"  TEST FOLDER  ({len(test_files)} images)")
print("=" * 55)
for filename in test_files:
    print(f"  {filename}")

print(f"\nEvaluation uses test/ images. Charts are saved to {metrics_dir}/.")
