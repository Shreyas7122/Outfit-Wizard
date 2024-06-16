from flask import Flask, request, redirect, url_for, session, render_template
import os
import cv2
import itertools
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.neighbors import NearestNeighbors

# Initialize the Flask application once
app = Flask(__name__, static_url_path='/static', static_folder='static')
app.secret_key = '1234'
dataset_dir = "static/dataset"
wardrobe_directory = "static/wardrobe"


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
    
def load_image(image_path):
    """Load an image from the specified path."""
    return cv2.imread(image_path)

def compute_histogram(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        print(f"Error loading image: {image_path}")
        return None

    # Convert the image to HSV color space
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Compute the histogram
    hist = cv2.calcHist([image_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    return hist

def compare_images(image1, image2):
    """Compare two images using their color histograms."""
    hist1 = compute_histogram(image1)
    hist2 = compute_histogram(image2)
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def get_image_paths(directory):
    """Get a list of all image paths in a directory."""
    return [os.path.join(directory, fname) for fname in os.listdir(directory) if fname.endswith(('.png', '.jpg', '.jpeg'))]

def find_most_similar_image(recommended_image_path, wardrobe_directory):
    best_match = None
    best_similarity = -float('inf')

    # Check if the directory exists
    if not os.path.exists(wardrobe_directory):
        print(f"Error: Wardrobe directory not found: {wardrobe_directory}")
        return best_match, best_similarity

    # Iterate over all images in the wardrobe directory
    for wardrobe_image in os.listdir(wardrobe_directory):
        wardrobe_image_path = os.path.join(wardrobe_directory, wardrobe_image)

        # Check if the file is an image
        if os.path.isfile(wardrobe_image_path) and wardrobe_image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            similarity = compare_images(recommended_image_path, wardrobe_image_path)
            
            # Check if the current similarity is the best so far
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = wardrobe_image_path

    return best_match, best_similarity



# Load ResNet model without the top (classification) layer
resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

def parse_folder_name(folder_name):
    parts = folder_name.split('_')
    occasion, gender, weather = parts[0], parts[1], parts[2]
    return occasion, gender, weather

def get_first_image_path(folder_path):
    files = os.listdir(folder_path)
    image_files = [f for f in files if f.endswith('.jpg') or f.endswith('.png')]
    if image_files:
        return os.path.join(folder_path, image_files[0])
    return None

import random

def recommend_images(occasion="formal", gender="men", weather="summer"):
    num_recommendations = 1  # Set the number of recommendations to 1 permanently
    # Load ResNet model without the top (classification) layer
    resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    def filter_images(dataset_dir, occasion=None, gender=None, weather=None):
        filtered_images = []
        if occasion is None or gender is None or weather is None:
            return filtered_images

        folder_name = f"{occasion}_{gender}_{weather}"
        folder_path = os.path.join(dataset_dir, folder_name)

        if os.path.exists(folder_path):
            image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]
            for image_file in image_files:
                image_path = os.path.join(folder_path, image_file)
                if os.path.isfile(image_path):
                    filtered_images.append(image_path)
        return filtered_images

    
    def preprocess_images(images):
        preprocessed_images = []
        for img_path in images:
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = preprocess_input(img_array)
            preprocessed_images.append(img_array)
        return np.array(preprocessed_images)
    
    # Filter images based on provided inputs or default preferences
    filtered_images = filter_images(dataset_dir, occasion, gender, weather)
    
    if not filtered_images:
        print("No images found for the specified criteria.")
        return []

    # Preprocess filtered images
    preprocessed_images = preprocess_images(filtered_images)
    
    # Extract features from images using the ResNet model
    features = resnet_model.predict(preprocessed_images, verbose=0)
    
    # Fit Nearest Neighbors model on the extracted features
    nn_model = NearestNeighbors(n_neighbors=num_recommendations, algorithm='auto')
    nn_model.fit(features.reshape(len(features), -1))
    
    # Randomly select a query image from the filtered images
    query_image_path = random.choice(filtered_images)
    
    # Load and preprocess the query image
    query_img = image.load_img(query_image_path, target_size=(224, 224))
    query_img_array = image.img_to_array(query_img)
    query_img_array = preprocess_input(query_img_array)

    # Extract features from the query image
    query_features = resnet_model.predict(np.expand_dims(query_img_array, axis=0), verbose=0)

    # Find nearest neighbors
    distances, indices = nn_model.kneighbors(query_features.reshape(1, -1))

    # Display recommended images
    recommended_images = [filtered_images[idx] for idx in indices[0]]
    return recommended_images

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/wardrobe.html')
def show_wardrobe():
    # Get the list of images in the wardrobe folder
    top_wear = 'static/split_outputs/top_wear'
    top_wear_images = []
    if os.path.exists(top_wear):
        top_wear_images = [os.path.join(top_wear, f) for f in os.listdir(top_wear) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Ensure all paths are correctly formatted for HTML
    top_wear_images = [img.replace("\\", "/") for img in top_wear_images]
    
    bottom_wear = 'static/split_outputs/bottom_wear'

    bottom_wear_images = []
    if os.path.exists(bottom_wear):
        bottom_wear_images = [os.path.join(bottom_wear, f) for f in os.listdir(bottom_wear) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Ensure all paths are correctly formatted for HTML
    bottom_wear_images = [img.replace("\\", "/") for img in bottom_wear_images]


    return render_template('wardrobe.html', top_wear_images=top_wear_images, bottom_wear_images=bottom_wear_images)

@app.route('/')
def homepage():
    return render_template('homepage.html')

@app.route('/input.html')
def input():
    return render_template('input.html')

@app.route('/upload.html')
def upload():
    return render_template('upload.html')

@app.route('/userdata', methods=['POST'])
def userdata():
    gender = request.form['gender']
    occasion = request.form['occasion']
    weather = request.form['weather']

    session['gender'] = gender
    session['occasion'] = occasion
    session['weather'] = weather

    recommended_images = recommend_images(occasion, gender, weather)
    session['recommended_images'] = recommended_images
    
    return redirect(url_for('output'))

@app.route('/output.html')
def output():
    recommended_images = session.get('recommended_images', [])

    # Ensure all paths in recommended_images are correctly formatted
    recommended_images = ['dataset/' + os.path.relpath(img, dataset_dir).replace("\\", "/") for img in recommended_images]
     
    wardrobe_directory = 'static/wardrobe'
    
    best_match, best_similarity = find_most_similar_image(recommended_images[0], wardrobe_directory)
    
    # Remove the redundant 'static/' from the path if present
    if best_match.startswith('static/'):
        best_match = best_match[len('static/'):]
    
    best_match = best_match.replace("\\", "/")
    print(best_similarity)
    
    return render_template('output.html', recommended_images=recommended_images, model2=best_match)


@app.route('/upload_1', methods=['POST'])
def upload_1():
    if 'images[]' not in request.files:
        return 'No file part'

    images = request.files.getlist('images[]')

    for image in images:
        if image.filename == '':
            return 'No selected file'
        if image:
            filename = image.filename
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return render_template('wardrobe.html')

def upload_2():
    input_folder = "static/uploads"
    split_output_folder = "static/split_outputs"
    top_output_folder = os.path.join(split_output_folder, "top_wear")
    bottom_output_folder = os.path.join(split_output_folder, "bottom_wear")
    wardrobe_folder = "static/wardrobe"
    top_wear_paths, bottom_wear_paths = process_folder(input_folder, top_output_folder, bottom_output_folder)
    create_combinations(top_wear_paths, bottom_wear_paths, wardrobe_folder)

    return render_template('wardrobe.html')

if __name__ == '__main__':
    app.run(debug=True)