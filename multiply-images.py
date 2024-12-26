from PIL import Image
import os
import pandas as pd

currnet_data = pd.read_csv('current_data.csv')
print("./images/messi.png")
final_data = pd.DataFrame(columns=currnet_data.columns)

# set current directory and final directory
directory = "./images"
final_directory = "./final_images"

# check if the final directory exists, if not create it
if not os.path.exists(final_directory):
    os.makedirs(final_directory)

new_data_line=0

def Add_Series(image, value, line_number):
    final_data.loc[line_number, "image"] = image
    final_data.loc[line_number, "value"] = value
    return line_number + 1

for image in currnet_data["image"]:
    if os.path.isfile(image):
        with Image.open(image) as img:
            image_value = currnet_data.loc[currnet_data["image"]==image, "value"].values[0]

            # save the original image
            imgSaveDir = os.path.join(final_directory, os.path.basename(image))
            img.save(imgSaveDir)
            name = image.split('/')[-1]
            new_data_line = Add_Series(f"./final_images/{name}", image_value, new_data_line)

            # Rotate the image, and flip it horizontally and vertically, and save them in the final directory
            for j in range(1, 3):
                rotated_image = img.rotate(90 * j)

                # Save the rotated image
                new_name = f"{os.path.splitext(os.path.basename(image))[0]}_rotated_{90 * j}{os.path.splitext(os.path.basename(image))[1]}"
                imgSaveDir = os.path.join(final_directory, new_name)
                rotated_image.save(imgSaveDir)
                new_data_line = Add_Series(f"./final_images/{new_name}", image_value, new_data_line)
                
                # Flip horizontally
                flipped_horizontally = rotated_image.transpose(Image.FLIP_LEFT_RIGHT)
                new_name_h = f"{os.path.splitext(os.path.basename(image))[0]}_rotated_{90 * j}_flipped_h{os.path.splitext(os.path.basename(image))[1]}"
                imgSaveDir_h = os.path.join(final_directory, new_name_h)
                flipped_horizontally.save(imgSaveDir_h)
                new_data_line = Add_Series(f"./final_images/{new_name_h}", image_value, new_data_line)
                
                # Flip vertically
                flipped_vertically = rotated_image.transpose(Image.FLIP_TOP_BOTTOM)
                new_name_v = f"{os.path.splitext(os.path.basename(image))[0]}_rotated_{90 * j}_flipped_v{os.path.splitext(os.path.basename(image))[1]}"
                imgSaveDir_v = os.path.join(final_directory, new_name_v)
                flipped_vertically.save(imgSaveDir_v)
                new_data_line = Add_Series(f"./final_images/{new_name_v}", image_value, new_data_line)

                if j != 2:
                    # Flip both horizontally and vertically
                    flipped_both = rotated_image.transpose(Image.ROTATE_180)
                    new_name_b = f"{os.path.splitext(os.path.basename(image))[0]}_rotated_{90 * j}_flipped_b{os.path.splitext(os.path.basename(image))[1]}"
                    imgSaveDir_b = os.path.join(final_directory, new_name_b)
                    flipped_both.save(imgSaveDir_b)
                    new_data_line = Add_Series(f"./final_images/{new_name_b}", image_value, new_data_line)
    else:
        print(f"\n{image} image does not exist.")

print(final_data)
final_data.to_csv("final_data.csv")