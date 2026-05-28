import argparse
import os
import random
import shutil

from model_config import (
    IMAGE_EXTENSIONS,
    MODEL_VARIANT,
    SPLIT_SEED,
    TEST_FOLDER_NAME,
    TRAIN_FOLDER_NAME,
    TRAIN_RATIO,
    VALIDATION_FOLDER_NAME,
    VALIDATION_RATIO,
    get_images_folder,
)


def is_image_file(filename):
    return filename.lower().endswith(IMAGE_EXTENSIONS)


def list_images_in_folder(folder):
    if not os.path.isdir(folder):
        return []
    images = []
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path) and is_image_file(filename):
            images.append(filename)
    return sorted(images)


def list_root_images(source_directory):
    images = []
    for filename in os.listdir(source_directory):
        file_path = os.path.join(source_directory, filename)
        if os.path.isfile(file_path) and is_image_file(filename):
            images.append(filename)
    return sorted(images)


def destination_exists(train_directory, validation_directory, test_directory, filename):
    return (
        os.path.exists(os.path.join(train_directory, filename))
        or os.path.exists(os.path.join(validation_directory, filename))
        or os.path.exists(os.path.join(test_directory, filename))
    )


def move_image(source_directory, destination_directory, filename):
    source_path = os.path.join(source_directory, filename)
    destination_path = os.path.join(destination_directory, filename)
    if not os.path.exists(source_path):
        return False
    if os.path.exists(destination_path):
        return False

    shutil.move(source_path, destination_path)
    return True


def split_counts(total_images):
    train_count = int(total_images * TRAIN_RATIO)
    validation_count = int(total_images * VALIDATION_RATIO)
    test_count = total_images - train_count - validation_count

    if total_images >= 3:
        if train_count < 1:
            train_count = 1
        if validation_count < 1:
            validation_count = 1
        if test_count < 1:
            test_count = 1

        while train_count + validation_count + test_count > total_images:
            if train_count >= validation_count and train_count >= test_count and train_count > 1:
                train_count -= 1
            elif validation_count >= test_count and validation_count > 1:
                validation_count -= 1
            elif test_count > 1:
                test_count -= 1
            else:
                break

        while train_count + validation_count + test_count < total_images:
            train_count += 1

    return train_count, validation_count, test_count


def consolidate_to_root(source_directory, train_directory, validation_directory, test_directory):
    moved_back = 0
    skipped = 0

    for folder_name, folder_path in (
        (TRAIN_FOLDER_NAME, train_directory),
        (VALIDATION_FOLDER_NAME, validation_directory),
        (TEST_FOLDER_NAME, test_directory),
    ):
        for filename in list_images_in_folder(folder_path):
            destination_path = os.path.join(source_directory, filename)
            source_path = os.path.join(folder_path, filename)
            if os.path.exists(destination_path):
                print(
                    f"Skipped {filename} from {folder_name}/ "
                    f"(same filename already exists in dataset root)."
                )
                skipped += 1
                continue

            shutil.move(source_path, destination_path)
            moved_back += 1

    return moved_back, skipped


def split_root_images(
    source_directory,
    train_directory,
    validation_directory,
    test_directory,
    root_images,
):
    if not root_images:
        return 0, 0, 0, 0, 0, 0

    random.seed(SPLIT_SEED)
    shuffled_images = list(root_images)
    random.shuffle(shuffled_images)

    training_size, validation_size, test_size = split_counts(len(shuffled_images))
    train_end = training_size
    validation_end = training_size + validation_size

    training_images = shuffled_images[:train_end]
    validation_images = shuffled_images[train_end:validation_end]
    test_images = shuffled_images[validation_end:]

    moved_training = 0
    moved_validation = 0
    moved_test = 0

    for filename in training_images:
        if move_image(source_directory, train_directory, filename):
            moved_training += 1

    for filename in validation_images:
        if move_image(source_directory, validation_directory, filename):
            moved_validation += 1

    for filename in test_images:
        if move_image(source_directory, test_directory, filename):
            moved_test += 1

    return (
        moved_training,
        moved_validation,
        moved_test,
        training_size,
        validation_size,
        test_size,
    )


def run_split(rebalance=False):
    source_directory = get_images_folder(MODEL_VARIANT)
    train_directory = os.path.join(source_directory, TRAIN_FOLDER_NAME)
    validation_directory = os.path.join(source_directory, VALIDATION_FOLDER_NAME)
    test_directory = os.path.join(source_directory, TEST_FOLDER_NAME)

    os.makedirs(train_directory, exist_ok=True)
    os.makedirs(validation_directory, exist_ok=True)
    os.makedirs(test_directory, exist_ok=True)

    existing_train_count = len(list_images_in_folder(train_directory))
    existing_validation_count = len(list_images_in_folder(validation_directory))
    existing_test_count = len(list_images_in_folder(test_directory))

    print(f"Using MODEL_VARIANT='{MODEL_VARIANT}' from model_config.py")
    print(
        f"Splitting dataset in {source_directory} into "
        f"{TRAIN_FOLDER_NAME}/, {VALIDATION_FOLDER_NAME}/, and {TEST_FOLDER_NAME}/..."
    )
    print(
        f"Already assigned: {existing_train_count} training, "
        f"{existing_validation_count} validation, {existing_test_count} test."
    )

    if rebalance:
        print("\nRebalance mode: moving all images from subfolders back to root...")
        moved_back, skipped = consolidate_to_root(
            source_directory,
            train_directory,
            validation_directory,
            test_directory,
        )
        print(f"Moved {moved_back} images back to root.")
        if skipped:
            print(f"Skipped {skipped} images due to filename conflicts in root.")

    if rebalance:
        root_images = list_root_images(source_directory)
        if not root_images:
            print("No images found to rebalance.")
        else:
            print(f"Re-splitting {len(root_images)} images using 60/20/20...")
            (
                moved_training,
                moved_validation,
                moved_test,
                training_size,
                validation_size,
                test_size,
            ) = split_root_images(
                source_directory,
                train_directory,
                validation_directory,
                test_directory,
                root_images,
            )
            print(
                f"Moved {moved_training} to {TRAIN_FOLDER_NAME}/, "
                f"{moved_validation} to {VALIDATION_FOLDER_NAME}/, "
                f"{moved_test} to {TEST_FOLDER_NAME}/."
            )
            print(
                f"Rebalance used 60/20/20 on {len(root_images)} images "
                f"({training_size} train, {validation_size} validation, {test_size} test)."
            )
    else:
        root_images = [
            filename
            for filename in list_root_images(source_directory)
            if not destination_exists(
                train_directory, validation_directory, test_directory, filename
            )
        ]

        if not root_images:
            print("No new root-level images to split.")
            print(
                "To convert an existing 80/20 split into 60/20/20, run:\n"
                "python split-train-validation.py --rebalance"
            )
            print(
                f"Current totals: {existing_train_count} training, "
                f"{existing_validation_count} validation, {existing_test_count} test."
            )
        else:
            (
                moved_training,
                moved_validation,
                moved_test,
                training_size,
                validation_size,
                test_size,
            ) = split_root_images(
                source_directory,
                train_directory,
                validation_directory,
                test_directory,
                root_images,
            )
            print(
                f"Moved {moved_training} to {TRAIN_FOLDER_NAME}/, "
                f"{moved_validation} to {VALIDATION_FOLDER_NAME}/, "
                f"{moved_test} to {TEST_FOLDER_NAME}/."
            )
            print(
                f"New split batch used 60/20/20 on {len(root_images)} root images "
                f"({training_size} train, {validation_size} validation, {test_size} test)."
            )

    final_train_count = len(list_images_in_folder(train_directory))
    final_validation_count = len(list_images_in_folder(validation_directory))
    final_test_count = len(list_images_in_folder(test_directory))
    print(
        f"Finished. Totals now: {final_train_count} training, "
        f"{final_validation_count} validation, {final_test_count} test."
    )
    print("Evaluation metrics/charts are saved separately in eval_images folders.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Split augmented images into train/, validation/, and test/ subfolders."
        )
    )
    parser.add_argument(
        "--rebalance",
        action="store_true",
        help=(
            "Move all images from train/, validation/, and test/ back to the dataset "
            "root, then re-split everything into a fresh 60/20/20 split."
        ),
    )
    args = parser.parse_args()
    run_split(rebalance=args.rebalance)


if __name__ == "__main__":
    main()
