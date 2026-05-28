from PIL import Image
import os

# Set the source directory and the separate directory for 5-degree augmented images.
directory = "./data/images"
final_directory = "./data/final_images_5deg"
image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif")

# Create the final directory if it doesn't exist.
if not os.path.exists(final_directory):
    os.makedirs(final_directory)

saved_count = 0
skipped_count = 0
repaired_count = 0
completed_count = 0
valid_existing_paths = set()


def image_is_valid(image_path):
    if not os.path.exists(image_path):
        return False

    try:
        with Image.open(image_path) as existing_image:
            existing_image.verify()
        return True
    except Exception:
        return False


def expected_output_names(filename):
    name, extension = os.path.splitext(filename)
    output_names = []

    for angle in range(0, 360, 5):
        output_names.append(f"{name}-rotated_{angle}{extension}")
        output_names.append(f"{name}-rotated_{angle}_flipped_h{extension}")

    return output_names


def print_resume_status():
    global completed_count, valid_existing_paths

    expected_paths = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(image_extensions):
            for output_name in expected_output_names(filename):
                expected_paths.append(os.path.join(final_directory, output_name))

    if not expected_paths:
        print("No source images found to augment.")
        return

    already_generated = 0
    last_image = None
    continuing_from = None

    for expected_path in expected_paths:
        if image_is_valid(expected_path):
            valid_existing_paths.add(expected_path)
            already_generated += 1
            last_image = os.path.basename(expected_path)
        else:
            continuing_from = os.path.basename(expected_path)
            break

    if continuing_from:
        print(
            f"{already_generated} images already generated. "
            f"Continuing from {continuing_from}."
        )
        if last_image:
            print(f"Last completed image: {last_image}")
    else:
        print(f"All {already_generated} expected images are already generated.")

    completed_count = already_generated


def save_augmented_image(image, save_path):
    global completed_count, saved_count, skipped_count, repaired_count

    if save_path in valid_existing_paths:
        skipped_count += 1
        return

    if image_is_valid(save_path):
        valid_existing_paths.add(save_path)
        completed_count += 1
        skipped_count += 1
        return

    if os.path.exists(save_path):
        repaired_count += 1

    name, extension = os.path.splitext(save_path)
    temp_path = f"{name}.tmp{extension}"
    image.save(temp_path)
    os.replace(temp_path, save_path)
    completed_count += 1
    saved_count += 1
    if completed_count % 50 == 0:
        print(f"Saved/confirmed {completed_count} augmented images so far...")


def output_is_done(save_path):
    global completed_count

    if save_path in valid_existing_paths:
        return True

    if image_is_valid(save_path):
        valid_existing_paths.add(save_path)
        completed_count += 1
        return True

    return False


print(f"Starting 5-degree augmentation from {directory} into {final_directory}...")
print_resume_status()

# Loop over all files in the source directory.
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)

    # Process only files that are images (you can add or remove extensions as needed)
    if os.path.isfile(file_path) and filename.lower().endswith(image_extensions):
        with Image.open(file_path) as img:
            name, extension = os.path.splitext(filename)

            # Unique 5-degree transforms: all rotations plus one reflected version per angle.
            for angle in range(0, 360, 5):
                rotated_name = f"{name}-rotated_{angle}{extension}"
                rotated_path = os.path.join(final_directory, rotated_name)
                flipped_h_name = f"{name}-rotated_{angle}_flipped_h{extension}"
                flipped_h_path = os.path.join(final_directory, flipped_h_name)

                if output_is_done(rotated_path) and output_is_done(flipped_h_path):
                    skipped_count += 2
                    continue

                rotated_image = img.rotate(angle)
                save_augmented_image(rotated_image, rotated_path)

                flipped_h = rotated_image.transpose(Image.FLIP_LEFT_RIGHT)
                save_augmented_image(flipped_h, flipped_h_path)
    else:
        continue

print(
    f"Finished 5-degree augmentation. Saved/confirmed {completed_count} total images, "
    f"saved {saved_count} new images, "
    f"skipped {skipped_count} existing images, repaired {repaired_count} broken images "
    f"into {final_directory}."
)
