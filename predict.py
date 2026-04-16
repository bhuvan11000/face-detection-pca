"""
Face Recognition Prediction Script.

This script takes a single input image and uses a pre-trained PCA model to 
identify the person in the image. 

The recognition process involves:
1. Loading the trained model (Eigenvectors, mean face, training signatures).
2. Pre-processing the input image (flattening, mean-subtraction).
3. Projecting the input image into 'Face Space' to get its signature.
4. Calculating the Euclidean distance between the test signature and all training signatures.
5. Using a softmax-like probability distribution to determine the most likely subject.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys

# Constants
MODEL_FILE = 'model.npz'
BASE_TRAIN_PATH = os.path.join('dataset', 'training')

def recognize_face(image_path, EigenVectors, W_train, mean_face,
                   NormsEigenVectors, m, n):
    """
    Main recognition logic.
    
    Args:
        image_path (str): Path to the test image (.pgm).
        EigenVectors (ndarray): The PCA basis (K x N_pixels).
        W_train (ndarray): Signatures of all training images (K x N_images).
        mean_face (ndarray): The average face from training (N_pixels,).
        NormsEigenVectors (ndarray): Diagonal elements of E.T @ E.
        m, n (int): Height and width of images.
        
    Returns:
        predicted_subject (int): The ID of the predicted person (1-40).
        probs_40 (ndarray): Probability distribution across the 40 subjects.
        distances (ndarray): Raw Euclidean distances to all 360 training images.
    """
    # 1. Load and pre-process the input image.
    img = np.array(Image.open(image_path)).astype(np.float64)
    x_test = img.reshape(m * n, 1) # Flatten to column vector.
    x_test -= mean_face.reshape(-1, 1) # Center the data using training mean.

    # 2. Project the test image onto the Eigenface basis.
    # w_test is the 'signature' of the input face.
    w_test = EigenVectors.T @ x_test / NormsEigenVectors.reshape(-1, 1)

    # 3. Calculate Euclidean distance between test signature and ALL training signatures.
    # distances will have 360 values (one for each training image).
    distances = np.linalg.norm(W_train - w_test, axis=0)
    
    # 4. Convert distances to probabilities using a Softmax-like function.
    # Shorter distance = Higher probability.
    s = -distances / (np.std(distances) + 1e-15) # Normalize distances.
    s -= s.max() # Numerical stability for exp().
    probs_360 = np.exp(s) / (np.exp(s).sum() + 1e-15)

    # 5. Aggregate probabilities for each subject.
    # Since each subject has 9 images, we take the max probability among their 9 images.
    n_subjects = 40
    imgs_per_subject = 9
    probs_40 = np.array([
        probs_360[i * imgs_per_subject : (i+1) * imgs_per_subject].max()
        for i in range(n_subjects)
    ])
    
    # Normalize the 40 probabilities to sum to 1.
    probs_40 /= (probs_40.sum() + 1e-15)
    
    # The predicted subject is the index with the highest probability (+1 for 1-based indexing).
    predicted_subject = int(probs_40.argmax()) + 1

    return predicted_subject, probs_40, distances

# --- Main Entry Point ---

# Ensure correct usage via command line.
if len(sys.argv) < 2:
    print("Usage: python3 predict.py <path_to_image.pgm>")
    exit(1)

image_path = sys.argv[1]
if not os.path.exists(image_path):
    print(f"Error: File '{image_path}' not found.")
    exit(1)

# Check if the model has been trained.
if not os.path.exists(MODEL_FILE):
    print(f"Model file '{MODEL_FILE}' not found. Please run train.py first.")
    exit(1)

# Load the model parameters.
data = np.load(MODEL_FILE)
EigenVectors = data['EigenVectors']
W_train = data['W_train']
mean_face = data['mean_face']
NormsEigenVectors = data['NormsEigenVectors']
m = int(data['m'])
n = int(data['n'])

# Run the recognition function.
predicted_subject, probs_40, distances = recognize_face(
    image_path, EigenVectors, W_train, mean_face, 
    NormsEigenVectors, m, n
)

# Output results to terminal.
confidence = probs_40[predicted_subject - 1] * 100
print(f"Prediction for '{image_path}':")
print(f"Subject: {predicted_subject:02d}")
print(f"Confidence: {confidence:.2f}%")

# --- Visualization ---

plt.figure(figsize=(8, 4))

# Subplot 1: The Input Image.
plt.subplot(1, 2, 1)
plt.imshow(np.array(Image.open(image_path)), cmap='gray')
plt.title(f"Input Image")
plt.axis('off')

# Subplot 2: The most similar image found in the training set.
plt.subplot(1, 2, 2)
best_idx = np.argmin(distances) # Index of the training image with minimum distance.
best_person = (best_idx // 9) + 1
best_img_num = (best_idx % 9) + 1
match_img_path = os.path.join(BASE_TRAIN_PATH, f'person_{best_person}', f'{best_img_num}.pgm')

plt.imshow(np.array(Image.open(match_img_path)), cmap='gray')
plt.title(f"Closest Match: P{best_person}")
plt.axis('off')

plt.tight_layout()
plt.show()
plt.close('all')
