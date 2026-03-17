import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys

MODEL_FILE = 'model.npz'
BASE_TRAIN_PATH = os.path.join('dataset', 'training')

def recognize_face(image_path, EigenVectors, W_train, mean_face,
                   NormsEigenVectors, m, n):
    img = np.array(Image.open(image_path)).astype(np.float64)
    x_test = img.reshape(m * n, 1)
    x_test -= mean_face.reshape(-1, 1)

    w_test = EigenVectors.T @ x_test / NormsEigenVectors.reshape(-1, 1)
    distances = np.linalg.norm(W_train - w_test, axis=0)
    
    s = -distances / (np.std(distances) + 1e-15)
    s -= s.max()
    probs_360 = np.exp(s) / (np.exp(s).sum() + 1e-15)

    n_subjects = 40
    imgs_per_subject = 9
    probs_40 = np.array([
        probs_360[i * imgs_per_subject : (i+1) * imgs_per_subject].max()
        for i in range(n_subjects)
    ])
    probs_40 /= (probs_40.sum() + 1e-15)
    predicted_subject = int(probs_40.argmax()) + 1

    return predicted_subject, probs_40, distances

if len(sys.argv) < 2:
    print("Usage: python3 predict.py <path_to_image.pgm>")
    exit(1)

image_path = sys.argv[1]
if not os.path.exists(image_path):
    print(f"Error: File '{image_path}' not found.")
    exit(1)

if not os.path.exists(MODEL_FILE):
    print(f"Model file '{MODEL_FILE}' not found. Please run train.py first.")
    exit(1)

# Load model
data = np.load(MODEL_FILE)
EigenVectors = data['EigenVectors']
W_train = data['W_train']
mean_face = data['mean_face']
NormsEigenVectors = data['NormsEigenVectors']
m = int(data['m'])
n = int(data['n'])

# Predict
predicted_subject, probs_40, distances = recognize_face(
    image_path, EigenVectors, W_train, mean_face, 
    NormsEigenVectors, m, n
)

confidence = probs_40[predicted_subject - 1] * 100

print(f"Prediction for '{image_path}':")
print(f"Subject: {predicted_subject:02d}")
print(f"Confidence: {confidence:.2f}%")

# Visualize
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(np.array(Image.open(image_path)), cmap='gray')
plt.title(f"Input Image")
plt.axis('off')

plt.subplot(1, 2, 2)
best_idx = np.argmin(distances)
best_person = (best_idx // 9) + 1
best_img_num = (best_idx % 9) + 1
match_img_path = os.path.join(BASE_TRAIN_PATH, f'person_{best_person}', f'{best_img_num}.pgm')
plt.imshow(np.array(Image.open(match_img_path)), cmap='gray')
plt.title(f"Closest Match: P{best_person}")
plt.axis('off')

plt.tight_layout()
plt.show()
plt.close('all')
