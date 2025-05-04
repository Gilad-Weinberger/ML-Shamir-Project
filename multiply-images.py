from PIL import Image
import os

# Set the source directory and the directory to save the augmented images.
directory = "./images"
final_directory = "./website/data/final_images"

# Create the final directory if it doesn't exist.
if not os.path.exists(final_directory):
    os.makedirs(final_directory)

# Loop over all files in the source directory.
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    
    # Process only files that are images (you can add or remove extensions as needed)
    if os.path.isfile(file_path) and filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
        # Open the image.
        with Image.open(file_path) as img:
            # Save the original image to the final directory.
            save_name = filename
            save_path = os.path.join(final_directory, save_name)
            img.save(save_path)

            # Apply augmentations: rotate, flip horizontally, vertically, and (if applicable) both.
            for j in range(1, 3):
                angle = 90 * j
                rotated_image = img.rotate(angle)
                
                # Save the rotated image.
                new_name = f"{os.path.splitext(filename)[0]}-rotated_{angle}{os.path.splitext(filename)[1]}"
                rotated_save_path = os.path.join(final_directory, new_name)
                rotated_image.save(rotated_save_path)
                
                # Flip horizontally.
                flipped_h = rotated_image.transpose(Image.FLIP_LEFT_RIGHT)
                new_name_h = f"{os.path.splitext(filename)[0]}-rotated_{angle}_flipped_h{os.path.splitext(filename)[1]}"
                flipped_h_path = os.path.join(final_directory, new_name_h)
                flipped_h.save(flipped_h_path)
                
                # Flip vertically.
                flipped_v = rotated_image.transpose(Image.FLIP_TOP_BOTTOM)
                new_name_v = f"{os.path.splitext(filename)[0]}-rotated_{angle}_flipped_v{os.path.splitext(filename)[1]}"
                flipped_v_path = os.path.join(final_directory, new_name_v)
                flipped_v.save(flipped_v_path)
                
                # For the first rotation (90°), also flip both horizontally and vertically.
                if j != 2:
                    flipped_b = rotated_image.transpose(Image.ROTATE_180)
                    new_name_b = f"{os.path.splitext(filename)[0]}-rotated_{angle}_flipped_b{os.path.splitext(filename)[1]}"
                    flipped_b_path = os.path.join(final_directory, new_name_b)
                    flipped_b.save(flipped_b_path)
    else:
        continue
