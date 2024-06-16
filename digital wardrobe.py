import cv2
import os
import itertools

def split_outfit(image_path, top_output_dir, bottom_output_dir):
    # Check if the image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None, None

    # Load the image
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        print(f"Error loading image: {image_path}")
        return None, None

    # Get image dimensions
    height, width, _ = image.shape

    # Calculate the midpoint of the height
    mid_height = height // 2

    # Split the image into top wear and bottom wear
    top_wear = image[:mid_height, :]
    bottom_wear = image[mid_height:, :]

    # Generate file names for the split images
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    top_wear_path = os.path.join(top_output_dir, f"{name}_top{ext}")
    bottom_wear_path = os.path.join(bottom_output_dir, f"{name}_bottom{ext}")

    # Save the top wear and bottom wear images
    cv2.imwrite(top_wear_path, top_wear)
    cv2.imwrite(bottom_wear_path, bottom_wear)

    return top_wear_path, bottom_wear_path

def process_folder(input_folder, top_output_folder, bottom_output_folder):
    # Check if the input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder not found: {input_folder}")
        return [], []

    # Create the output folders if they do not exist
    if not os.path.exists(top_output_folder):
        os.makedirs(top_output_folder)
    if not os.path.exists(bottom_output_folder):
        os.makedirs(bottom_output_folder)

    top_wear_paths = []
    bottom_wear_paths = []

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            top_wear_path, bottom_wear_path = split_outfit(file_path, top_output_folder, bottom_output_folder)
            if top_wear_path and bottom_wear_path:
                top_wear_paths.append(top_wear_path)
                bottom_wear_paths.append(bottom_wear_path)

    return top_wear_paths, bottom_wear_paths

def resize_image_to_width(image, width):
    # Calculate the aspect ratio
    ratio = width / image.shape[1]
    new_dimensions = (width, int(image.shape[0] * ratio))

    # Resize the image
    resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
    return resized_image

def create_combinations(top_wear_paths, bottom_wear_paths, output_folder):
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    combination_counter = 0

    # Generate all combinations of top wear and bottom wear
    for top_wear_path, bottom_wear_path in itertools.product(top_wear_paths, bottom_wear_paths):
        # Load the top wear and bottom wear images
        top_wear = cv2.imread(top_wear_path)
        bottom_wear = cv2.imread(bottom_wear_path)

        # Ensure the images have the same width
        width = min(top_wear.shape[1], bottom_wear.shape[1])
        top_wear = resize_image_to_width(top_wear, width)
        bottom_wear = resize_image_to_width(bottom_wear, width)

        # Concatenate the images vertically
        combined_image = cv2.vconcat([top_wear, bottom_wear])

        # Save the combined image
        combination_filename = f"combination_{combination_counter:04d}.png"
        combination_path = os.path.join(output_folder, combination_filename)
        cv2.imwrite(combination_path, combined_image)
        
        combination_counter += 1

    print(f"Created {combination_counter} combinations.")

# Example usage
input_folder = "static/uploads"
split_output_folder = "static/split_outputs"
top_output_folder = os.path.join(split_output_folder, "top_wear")
bottom_output_folder = os.path.join(split_output_folder, "bottom_wear")
wardrobe_folder = "static/wardrobe"

top_wear_paths, bottom_wear_paths = process_folder(input_folder, top_output_folder, bottom_output_folder)
create_combinations(top_wear_paths, bottom_wear_paths, wardrobe_folder)
