"""
Face Recognition Evaluation Script.

This script evaluates the performance of the PCA-based face recognition system 
on a separate test dataset. 

Key features:
1. Calculates overall classification accuracy on 40 test images.
2. Identifies misclassified subjects.
3. Performs a 'Top-k Sweep' to see how the number of used eigenfaces affects accuracy.
4. Visualizes top predictions and probability bars for each test case.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Configuration: Paths and files.
TEST_PATH = os.path.join('dataset', 'testing')
BASE_TRAIN_PATH = os.path.join('dataset', 'training')
MODEL_FILE = 'model.npz'

def recognize_face(image_path, EigenVectors, W_train, mean_face,
                   NormsEigenVectors, m, n, top_k=None):
    """
    Recognition logic used during testing.
    Includes an optional 'top_k' parameter to limit the number of eigenfaces used.
    """
    # Load and preprocess.
    img = np.array(Image.open(image_path)).astype(np.float64)
    x_test = img.reshape(m * n, 1)
    x_test -= mean_face.reshape(-1, 1)

    # Optional: Use only a subset (top_k) of the principal components.
    if top_k is not None:
        E = EigenVectors[:, :top_k]
        norms = NormsEigenVectors[:top_k]
        W = W_train[:top_k, :]
    else:
        E = EigenVectors
        norms = NormsEigenVectors
        W = W_train

    # Project and compute distances in reduced feature space.
    w_test = E.T @ x_test / norms.reshape(-1, 1)
    distances = np.linalg.norm(W - w_test, axis=0)
    
    # Probability conversion (Softmax-like).
    s = -distances / (np.std(distances) + 1e-15)
    s -= s.max()
    probs_360 = np.exp(s) / (np.exp(s).sum() + 1e-15)

    # Subject aggregation.
    n_subjects = 40
    imgs_per_subject = 9
    probs_40 = np.array([
        probs_360[i * imgs_per_subject : (i+1) * imgs_per_subject].max()
        for i in range(n_subjects)
    ])
    probs_40 /= (probs_40.sum() + 1e-15)
    predicted_subject = int(probs_40.argmax()) + 1

    return predicted_subject, probs_40, distances

# --- Initialize Environment ---

if not os.path.exists(MODEL_FILE):
    print(f"Model file '{MODEL_FILE}' not found. Please run train.py first.")
    exit(1)

# Load the trained model.
data = np.load(MODEL_FILE)
EigenVectors = data['EigenVectors']
W_train = data['W_train']
mean_face = data['mean_face']
NormsEigenVectors = data['NormsEigenVectors']
m = int(data['m'])
n = int(data['n'])

# ============================================================
# RUN ACCURACY ON TEST SET
# ============================================================
"""
Iterate through the 40 test images (one for each person) and compare 
the prediction against the true identity.
"""
results = []
correct_count = 0
misidentified = []

print("-" * 30)
for i in range(1, 41):
    test_path = os.path.join(TEST_PATH, f'p_{i}.pgm')
    predicted_subject, probs_40, distances = recognize_face(
        test_path, EigenVectors, W_train, mean_face, 
        NormsEigenVectors, m, n
    )
    
    confidence = probs_40[predicted_subject - 1] * 100
    correct = (predicted_subject == i)
    
    if correct:
        correct_count += 1
        mark = "✓"
    else:
        misidentified.append(i) # Store IDs of subjects that were misclassified.
        mark = "✗"
    
    print(f"p_{i}.pgm | True: {i:02d} | Pred: {predicted_subject:02d} {mark} | Confidence: {confidence:.1f}%")
    results.append((i, predicted_subject, probs_40, distances))

accuracy = (correct_count / 40) * 100
print("-" * 30)
print(f"Overall Accuracy: {correct_count} / 40 = {accuracy:.1f}%")
print(f"Misidentified Subject IDs: {misidentified if misidentified else 'None'}")

# ============================================================
# VISUALIZE TEST RESULTS
# ============================================================
"""
Display a detailed visual report for every test case.
Shows the input, the closest training match, and a probability bar chart.
"""
for i, predicted, probs_40, distances in results:
    plt.figure(figsize=(12, 4))
    
    # 1. Test Image: The input image being classified.
    plt.subplot(1, 3, 1)
    test_img = Image.open(os.path.join(TEST_PATH, f'p_{i}.pgm'))
    plt.imshow(np.array(test_img), cmap='gray')
    plt.title(f"Input: Person {i}")
    plt.axis('off')
    
    # 2. Closest Match: The image in the training database that was nearest in Face Space.
    plt.subplot(1, 3, 2)
    best_idx = np.argmin(distances)
    best_person = (best_idx // 9) + 1
    best_img_num = (best_idx % 9) + 1
    match_img_path = os.path.join(BASE_TRAIN_PATH, f'person_{best_person}', f'{best_img_num}.pgm')
    plt.imshow(np.array(Image.open(match_img_path)), cmap='gray')
    plt.title(f"Match: Person {best_person} {'✓' if predicted == i else '✗'}")
    plt.axis('off')
    
    # 3. Probabilities: Horizontal bar chart showing Top-5 candidate predictions.
    plt.subplot(1, 3, 3)
    top5_idx = np.argsort(probs_40)[-5:]
    top5_probs = probs_40[top5_idx]
    top5_labels = [f"P{idx+1}" for idx in top5_idx]
    
    # The true subject's bar is highlighted in green, others in red.
    colors = ['green' if (idx + 1) == i else 'red' for idx in top5_idx]
    
    plt.barh(top5_labels, top5_probs, color=colors)
    plt.xlabel('Probability')
    plt.title('Top-5 Probabilities')
    
    plt.tight_layout()
    plt.show()
    plt.close('all')

# ============================================================
# TOP_K SWEEP
# ============================================================
"""
Evaluate how accuracy changes as we vary the number of principal components (k).
This helps identify the minimum number of eigenfaces needed for high performance.
"""
k_values = [1, 5, 10, 20, 30, 50, 75, 100, 150, 200, 359]
k_accuracies = []

print("\nStarting Top-k Sweep...")
for k in k_values:
    k_correct = 0
    for i in range(1, 41):
        test_path = os.path.join(TEST_PATH, f'p_{i}.pgm')
        pred, _, _ = recognize_face(test_path, EigenVectors, W_train, mean_face, NormsEigenVectors, m, n, top_k=k)
        if pred == i: 
            k_correct += 1
    k_accuracies.append((k_correct / 40) * 100)

# Plot the accuracy curve for the Top-k Sweep.
plt.figure()
plt.plot(k_values, k_accuracies, 'bo-')
plt.title('Accuracy vs Number of Eigenfaces (k)')
plt.xlabel('k (eigenfaces used)')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.show()
plt.close('all')
