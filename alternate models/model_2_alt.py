import json
import os
import numpy as np
import requests
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.neighbors import NearestNeighbors

def preprocess_images(image_paths, img_height, img_width):
    images = []
    for img_path in image_paths:
        img = load_img(img_path, target_size=(img_height, img_width))
        img_array = img_to_array(img) / 255.0  # Normalize pixel values
        images.append(img_array)
    return np.array(images)

def load_filtered_images(image_paths, folder_names):
    labels = [folder_names.index(os.path.basename(os.path.dirname(img_path))) for img_path in image_paths]
    return preprocess_images(image_paths, img_height, img_width), np.array(labels)

def load_image_dataset(dataset_path, weather, occasion, gender):
    filtered_images = []
    folder_names = []
    for root, dirs, files in os.walk(dataset_path):
        folder_name = os.path.basename(root)
        if weather in folder_name.lower() and occasion in folder_name.lower() and gender in folder_name.lower():
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png"):
                    filtered_images.append(os.path.join(root, file))
            folder_names.append(folder_name)
    return filtered_images, folder_names

def load_preferences_from_flask():
    # Return default values
    return "winter", "casual", "men"

img_height, img_width = 224, 224

def load_dataset():
    dataset_path = "dataset"
    weather, occasion, gender = load_preferences_from_flask()
    filtered_images, folder_names = load_image_dataset(dataset_path, weather, occasion, gender)
    return filtered_images, folder_names

def build_cnn_model(img_height, img_width, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def main():
    filtered_images, folder_names = load_dataset()
    print("Number of images found:", len(filtered_images))
    if len(filtered_images) == 0:
        print("No matching images found. Exiting...")
        return

    X, y = load_filtered_images(filtered_images, folder_names)
    num_classes = len(folder_names)

    cnn_model = build_cnn_model(img_height, img_width, num_classes)
    cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cnn_model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

    features = cnn_model.predict(X)

    knn_model = NearestNeighbors(n_neighbors=5, metric='euclidean')
    knn_model.fit(features)
    query_feature = features[0].reshape(1, -1)
    distances, indices = knn_model.kneighbors(query_feature)
    recommended_outfits = [filtered_images[i] for i in indices[0]]

    # Save the model as "model_1"
  #  save_model(cnn_model, "model_1.h5")

    # Return the filename of the recommended outfit
    return recommended_outfits[0]

if __name__ == "__main__":
    recommended_outfit = main()
    print("Recommended outfit filename:", recommended_outfit)
