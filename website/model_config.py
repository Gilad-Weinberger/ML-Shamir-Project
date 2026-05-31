import os

# ============================================================
# MODEL VARIANT SELECTION
# Change ONLY this variable for training, splitting, and evaluation.
#
# "original"      -> grape_leaf_model.pth + data/final_images
# "original_test" -> grape_leaf_model_test.pth + data/final_images
# "test_2"        -> grape_leaf_model_test_2.pth + data/final_images
# "5deg"          -> grape_leaf_model_5deg.pth + data/final_images_5deg
#
# Each dataset is split into:
#   train/ validation/ test/
#
# Evaluation runs on test/ images and saves charts to evals/<variant>/.
# ============================================================
MODEL_VARIANT = "5deg"
EVALS_FOLDER = "evals"

MODEL_VARIANTS = {
    "original": {
        "model_file": "grape_leaf_model.pth",
        "images_folder": "data/final_images",
    },
    "original_test": {
        "model_file": "grape_leaf_model_test.pth",
        "images_folder": "data/final_images",
    },
    "test_2": {
        "model_file": "grape_leaf_model_test_2.pth",
        "images_folder": "data/final_images",
    },
    "5deg": {
        "model_file": "grape_leaf_model_5deg.pth",
        "images_folder": "data/final_images_5deg",
    },
}

TRAIN_FOLDER_NAME = "train"
VALIDATION_FOLDER_NAME = "validation"
TEST_FOLDER_NAME = "test"
TRAIN_RATIO = 0.6
VALIDATION_RATIO = 0.2
TEST_RATIO = 0.2
SPLIT_SEED = 42
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif")

NUM_EPOCHS = 60
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.001
EVALUATION_THRESHOLD = 5.0


def get_variant_config(variant=None):
    selected_variant = variant or MODEL_VARIANT
    if selected_variant not in MODEL_VARIANTS:
        valid_variants = ", ".join(MODEL_VARIANTS)
        raise ValueError(
            f"Unknown MODEL_VARIANT '{selected_variant}'. Use one of: {valid_variants}"
        )
    return MODEL_VARIANTS[selected_variant]


def get_model_file(variant=None):
    return get_variant_config(variant)["model_file"]


def get_images_folder(variant=None):
    return get_variant_config(variant)["images_folder"]


def get_eval_folder(variant=None):
    """Output folder for evaluation charts/metrics only."""
    selected_variant = variant or MODEL_VARIANT
    get_variant_config(selected_variant)
    return os.path.join(EVALS_FOLDER, selected_variant)


def get_train_folder(variant=None):
    return os.path.join(get_images_folder(variant), TRAIN_FOLDER_NAME)


def get_validation_folder(variant=None):
    return os.path.join(get_images_folder(variant), VALIDATION_FOLDER_NAME)


def get_test_folder(variant=None):
    return os.path.join(get_images_folder(variant), TEST_FOLDER_NAME)
