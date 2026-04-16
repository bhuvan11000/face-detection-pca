"""
PCA-based Face Recognition Training Script.

This script implements the Eigenfaces algorithm for face recognition. 
It performs the following steps:
1. Loads training images and converts them into a high-dimensional data matrix.
2. Computes the average (mean) face of the dataset.
3. Normalizes the data by subtracting the mean face.
4. Uses the 'P.T @ P' trick to efficiently compute high-dimensional eigenvectors.
5. Projects the training images onto the principal component space (Eigenfaces).
6. Saves the resulting model (eigenvectors, mean face, weights) for future use.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Configuration: Paths for the training dataset and the output model file.
BASE_PATH = os.path.join('dataset', 'training')
MODEL_FILE = 'model.npz'

# Check if a trained model already exists to avoid accidental overwrites.
if os.path.exists(MODEL_FILE):
    print(f"Model already exists in '{MODEL_FILE}'. Delete it to retrain.")
    exit(0)

# ============================================================
# TASK 1: Load training images and build matrix P
# ============================================================
"""
We iterate through the dataset, reading 9 images for each of the 40 subjects.
Each image is flattened into a long column vector.
The final matrix P will have dimensions (N_pixels, N_images).
"""

Database_Size = 40  # Total number of unique subjects in the training set.
images = []         # List to store flattened image vectors.
train_labels = []   # List to store ground-truth labels for training images.

m = 0               # Image height (to be determined from the first image).
n = 0               # Image width (to be determined from the first image).

print("Loading training images...")
for i in range(1, Database_Size + 1):
    for j in range(1, 10): # Using first 9 images for training, 10th is for testing.
        img_path = os.path.join(BASE_PATH, f'person_{i}', f'{j}.pgm')
        img = Image.open(img_path)
        img_array = np.array(img)
        
        # Capture dimensions from the first image encountered.
        if m == 0 and n == 0:
            m, n = img_array.shape
            
        # Flatten the 2D image (m x n) into a 1D column vector (m*n x 1).
        col_vector = img_array.reshape(m * n, 1)
        images.append(col_vector)
        train_labels.append(i)

# Concatenate all column vectors horizontally to create the data matrix P.
P = np.hstack(images)

print(f"Database_Size: {Database_Size}")
print(f"Image dimensions (m, n): ({m}, {n})")
print(f"P matrix shape: {P.shape}") # (10304, 360) for 112x92 images and 40*9 samples.

# ============================================================
# TASK 2: Compute and display the mean face
# ============================================================
"""
The mean face represents the 'average' features across all subjects.
Subtracting it helps isolate the unique features (variations) of each individual.
"""

# Calculate the average vector across all columns (images).
mean_face = np.mean(P, axis=1) # Shape: (N_pixels,)

# Display the mean face as a grayscale image.
plt.figure()
plt.imshow(mean_face.reshape(m, n), cmap='gray')
plt.title('Mean Face')
plt.axis('off')
plt.tight_layout()
plt.show()
plt.close('all')

# ============================================================
# TASK 3: Subtract mean face from every column of P
# ============================================================
"""
Center the data by subtracting the mean. This is a prerequisite for PCA.
We convert to float64 to maintain precision during subtraction.
"""

P = P.astype(np.float64)
# mean_face.reshape(-1, 1) ensures the 1D mean vector is broadcasted correctly across all columns of P.
P = P - mean_face.reshape(-1, 1)

# ============================================================
# TASK 4: Eigendecomposition using the PᵀP trick
# ============================================================
"""
Standard PCA requires finding eigenvectors of the Covariance Matrix (P @ P.T), 
which is 10304x10304. This is computationally expensive.
Instead, we find eigenvectors of (P.T @ P), which is only 360x360.
The high-dimensional eigenvectors are then recovered by multiplying P with the low-dimensional ones.
"""

print("Performing Eigendecomposition...")
# Compute the smaller matrix (N_images x N_images).
PTP = P.T @ P
# Calculate eigenvalues (Values) and eigenvectors (Vectors) of the smaller matrix.
Values, Vectors = np.linalg.eig(PTP)

# Sort eigenvalues and their corresponding eigenvectors in descending order.
# Large eigenvalues correspond to principal components that capture the most variance.
sort_idx = np.argsort(Values)[::-1]
Values = Values[sort_idx]
Vectors = Vectors[:, sort_idx]

# Recover the eigenvectors of the original high-dimensional space.
# Formula: EigenVectors_Large = P @ EigenVectors_Small
EigenVectors = P @ Vectors

# Normalize each recovered eigenvector to have unit length (magnitude of 1).
EigenVectors = EigenVectors / np.linalg.norm(EigenVectors, axis=0)

# Drop the last column if its eigenvalue is near zero (mathematical artifact).
# We keep 359 out of 360 components.
EigenVectors = EigenVectors[:, :359] 

print("Eigenvalues computed.")

# ============================================================
# TASK 5: Display the first 29 eigenfaces
# ============================================================
"""
Eigenfaces are the principal components visualized as images.
They represent the most significant variations found in the face database.
"""

eigenfaces_list = []
for j in range(29):
    # To visualize, we add the mean face back to the eigenvector and clip values to 0-255 range.
    ef = EigenVectors[:, j] + mean_face
    ef_img = ef.reshape(m, n).clip(0, 255).astype(np.uint8)
    eigenfaces_list.append(ef_img)

# Combine the first 29 eigenfaces into one large horizontal image for comparison.
EigenFaces = np.hstack(eigenfaces_list)

plt.figure(figsize=(15, 5))
plt.imshow(EigenFaces, cmap='gray')
plt.title('First 29 Eigenfaces')
plt.axis('off')
plt.tight_layout()
plt.show()
plt.close('all')

# ============================================================
# TASK 6: Verify orthogonality of eigenvectors
# ============================================================
"""
In PCA, eigenvectors should be orthogonal (independent).
Multiplying the transpose of the matrix by itself should result in an identity matrix
(or a diagonal matrix if not perfectly normalized).
"""

Products = EigenVectors.T @ EigenVectors
NormsEigenVectors = np.diag(Products) # Should be all 1s if normalized correctly.
is_diagonal = np.allclose(Products, np.diag(NormsEigenVectors))

print(f"Is Products matrix diagonal? {is_diagonal}")

# ============================================================
# TASK 7: Project all training images → face signatures W_train
# ============================================================
"""
Each image in the training set is projected into the 'Face Space'.
The result is a 'weight' vector (or signature) that represents how much of 
each eigenface is present in that specific image.
"""

# Project centered training data P onto the Eigenface basis.
W_train = EigenVectors.T @ P
# Divide by norms to ensure consistency if eigenvectors weren't perfectly unit-length.
W_train = W_train / NormsEigenVectors.reshape(-1, 1)

print(f"W_train shape: {W_train.shape}") # (359 components, 360 images)

# ============================================================
# TASK 8: Plot eigenvalue decay curve
# ============================================================
"""
This plot shows how much information (variance) each principal component carries.
Typically, the first few components carry most of the facial information.
"""

plt.figure()
plt.semilogy(range(1, 360), Values[:359]) # log scale highlights the decay rate.
plt.title('Eigenvalue Decay')
plt.xlabel('Principal Component Index')
plt.ylabel('Eigenvalue (log scale)')
plt.axvline(x=50, color='r', linestyle='--', label='top 50')
plt.legend()
plt.tight_layout()
plt.show()
plt.close('all')

# ============================================================
# SAVE MODEL
# ============================================================
"""
Save all necessary components to an .npz file so they can be loaded by the 
prediction script without retraining.
"""

np.savez(MODEL_FILE, 
         EigenVectors=EigenVectors,     # The principal component basis.
         W_train=W_train,               # Signatures of all training images.
         mean_face=mean_face,           # Needed to center new test images.
         NormsEigenVectors=NormsEigenVectors, 
         train_labels=np.array(train_labels), 
         m=m, n=n,                      # Original image dimensions.
         Values=Values)                 # Eigenvalues (variance captured).

print(f"Model saved to '{MODEL_FILE}'")
