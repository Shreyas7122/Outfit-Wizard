import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import random

dataset_dir = "dataset"
print("yes")  

# Ensure the dataset directory exists
if not os.path.exists(dataset_dir):
    print(f"The specified dataset directory '{dataset_dir}' does not exist.")

def parse_folder_name(folder_name):
    # Split folder name by underscores
    parts = folder_name.split('_')
    occasion = parts[0]
    gender = parts[1]
    weather = parts[2]
    return occasion, gender, weather

def get_first_image_path(folder_path):
    # Get list of files in the folder
    files = os.listdir(folder_path)
    
    # Filter image files (jpg or png)
    image_files = [f for f in files if f.endswith('.jpg') or f.endswith('.png')]
    
    if image_files:
        # Return path of the first image in the folder
        return os.path.join(folder_path, image_files[0])
    else:
        # Return None if no image found in the folder
        return None

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

# Example usage:
occasion = "formal"
gender = "men"
weather = "winter"  # Default preference for weather is 'summer'

recommended_images = recommend_images(occasion=occasion, gender=gender, weather=weather)

print("Recommended Image for occasion '{}', gender '{}', and weather '{}':".format(occasion, gender, weather))
for img_path in recommended_images:
    print(img_path)
    image = Image.open(img_path)
    image.show()
