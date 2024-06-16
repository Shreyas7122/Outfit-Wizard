import os
import cv2
import numpy as np

def load_image(image_path):
    """Load an image from the specified path."""
    return cv2.imread(image_path)

def compute_histogram(image):
    """Compute color histogram for an image."""
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([image_hsv], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
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
    """Find the most similar image in the wardrobe directory to the recommended image."""
    recommended_image = load_image(recommended_image_path)
    wardrobe_image_paths = get_image_paths(wardrobe_directory)

    best_match = None
    best_similarity = -1

    for wardrobe_image_path in wardrobe_image_paths:
        wardrobe_image = load_image(wardrobe_image_path)
        similarity = compare_images(recommended_image, wardrobe_image)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = wardrobe_image_path

    return best_match, best_similarity

# Usage
recommended_image_path = 'dataset/casual_men_winter/1.jpg'  # Replace with your image path
wardrobe_directory = 'uploads'
best_match, best_similarity = find_most_similar_image(recommended_image_path, wardrobe_directory)

print(f"Most similar image: {best_match}, Similarity: {best_similarity:.2f}")