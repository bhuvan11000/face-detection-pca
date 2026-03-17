# Face Detection using NumPy (Eigenfaces)

This project implements a face recognition system using the **Eigenface** algorithm, which leverages Principal Component Analysis (PCA) to represent faces as a linear combination of "eigenfaces." It is built entirely using NumPy, PIL (Pillow), and Matplotlib.

## Project Structure

```text
face-detection-numpy/
├── dataset/                # Face images (training and testing)
│   ├── training/           # 40 subjects, 9 images each (.pgm)
│   └── testing/            # 40 subjects, 1 image each (.pgm)
├── predict.py              # Script to predict a subject for a given image
├── train.py                # Script to train the model and generate eigenfaces
├── test_accuracy.py        # Script to evaluate model accuracy on the test set
├── model.npz               # Saved model (generated after training)
├── README.md               # Project documentation
└── .gitignore              # Files to ignore in version control
```

## How It Works

The project uses the **Eigenfaces** method:

1.  **Training (`train.py`):**
    *   **Data Loading:** Loads grayscale face images from the training dataset.
    *   **Mean Face:** Computes the average face of the entire dataset and subtracts it from each image to center the data.
    *   **PCA (Eigendecomposition):** Uses the $P^T P$ trick to efficiently find the eigenvectors (Eigenfaces) and eigenvalues of the covariance matrix.
    *   **Signature Extraction:** Projects each training image onto the "Face Space" defined by the top eigenvectors to create a "face signature" (weights).
    *   **Model Storage:** Saves the eigenvectors, mean face, and signatures into `model.npz`.

2.  **Prediction (`predict.py`):**
    *   Takes an input image, subtracts the mean face, and projects it into the same Face Space.
    *   Calculates the Euclidean distance between the input's signature and all training signatures.
    *   Identifies the subject with the minimum distance (closest match).

## Setup Instructions

### 1. Prerequisites
Ensure you have Python installed along with the required libraries:
```bash
pip install numpy pillow matplotlib
```

### 2. Download the Dataset
The dataset is hosted on Google Drive. 
1.  Go to the [Dataset Folder](https://drive.google.com/drive/folders/1qGdO7TrBOeL5t4PPsK42wodwd-xoPxoy?usp=drive_link).
2.  Download the **`dataset.zip`** file.
3.  Extract the contents of `dataset.zip` directly into the project root directory. Your folder structure should now include `dataset/training/` and `dataset/testing/`.

### 3. Training the Model
Run the training script to generate the model:
```bash
python3 train.py
```
This will display the mean face and the top eigenfaces, then save `model.npz`.

### 4. Running Predictions
To recognize a face from an image:
```bash
python3 predict.py dataset/testing/person_1/10.pgm
```

### 5. Testing Accuracy
To see how well the model performs across the entire test set:
```bash
python3 test_accuracy.py
```

## Features
*   **Pure NumPy implementation** for core linear algebra.
*   **Visualizations** of Mean Face, Eigenfaces, and Eigenvalue decay.
*   **Confidence estimation** based on distance distributions.
*   **Side-by-side comparison** of the input image and the closest match in the training set.
