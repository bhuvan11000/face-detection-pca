import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

BASE_PATH = os.path.join('dataset', 'training')
MODEL_FILE = 'model.npz'

if os.path.exists(MODEL_FILE):
    print(f"Model already exists in '{MODEL_FILE}'. Delete it to retrain.")
    exit(0)

# ============================================================
# TASK 1: Load training images and build matrix P
# ============================================================

Database_Size = 40
images = []
train_labels = []

m = 0
n = 0

print("Loading training images...")
for i in range(1, Database_Size + 1):
    for j in range(1, 10):
        img_path = os.path.join(BASE_PATH, f'person_{i}', f'{j}.pgm')
        img = Image.open(img_path)
        img_array = np.array(img)
        
        if m == 0 and n == 0:
            m, n = img_array.shape
            
        col_vector = img_array.reshape(m * n, 1)
        images.append(col_vector)
        train_labels.append(i)

P = np.hstack(images)

print(f"Database_Size: {Database_Size}")
print(f"Image dimensions (m, n): ({m}, {n})")
print(f"P matrix shape: {P.shape}")

# ============================================================
# TASK 2: Compute and display the mean face
# ============================================================

mean_face = np.mean(P, axis=1) # shape (10304,)

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

P = P.astype(np.float64)
P = P - mean_face.reshape(-1, 1)

# ============================================================
# TASK 4: Eigendecomposition using the PᵀP trick
# ============================================================

print("Performing Eigendecomposition...")
PTP = P.T @ P
Values, Vectors = np.linalg.eig(PTP)

sort_idx = np.argsort(Values)[::-1]
Values = Values[sort_idx]
Vectors = Vectors[:, sort_idx]

EigenVectors = P @ Vectors
EigenVectors = EigenVectors / np.linalg.norm(EigenVectors, axis=0)

# Drop the last column (zero eigenvalue)
EigenVectors = EigenVectors[:, :359] 

print("Eigenvalues computed.")

# ============================================================
# TASK 5: Display the first 29 eigenfaces
# ============================================================

eigenfaces_list = []
for j in range(29):
    ef = EigenVectors[:, j] + mean_face
    ef_img = ef.reshape(m, n).clip(0, 255).astype(np.uint8)
    eigenfaces_list.append(ef_img)

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

Products = EigenVectors.T @ EigenVectors
NormsEigenVectors = np.diag(Products)
is_diagonal = np.allclose(Products, np.diag(NormsEigenVectors))

print(f"Is Products matrix diagonal? {is_diagonal}")

# ============================================================
# TASK 7: Project all training images → face signatures W_train
# ============================================================

W_train = EigenVectors.T @ P
W_train = W_train / NormsEigenVectors.reshape(-1, 1)

print(f"W_train shape: {W_train.shape}")

# ============================================================
# TASK 8: Plot eigenvalue decay curve
# ============================================================

plt.figure()
plt.semilogy(range(1, 360), Values[:359])
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

np.savez(MODEL_FILE, 
         EigenVectors=EigenVectors, 
         W_train=W_train, 
         mean_face=mean_face, 
         NormsEigenVectors=NormsEigenVectors, 
         train_labels=np.array(train_labels), 
         m=m, n=n,
         Values=Values)

print(f"Model saved to '{MODEL_FILE}'")
