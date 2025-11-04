# Alzheimer's Classification Model Analysis

This repository contains several Jupyter Notebooks exploring different machine learning and deep learning models for classifying Alzheimer's disease from brain scan images.

---

## Notebook Analysis & Accuracy Summary

### 1. `SVM_Alzheimer.ipynb`

This notebook explores combining Convolutional Neural Networks (CNNs) for feature extraction with Support Vector Machine (SVM) classifiers.

* **VGG16 + Linear SVM:**
    * Validation Accuracy: 72.20%
    * Test Accuracy: 56.61%
* **VGG16 + RBF SVM:**
    * Validation Accuracy: 72.11%
    * Test Accuracy: 61.30%
* **VGG16 + Polynomial SVM (Degree 6):**
    * Train Accuracy: 92.36%
    * Test Accuracy: 65.44%
* **Custom CNN (Simple) + SVM:**
    * RBF Kernel Accuracy: 50.43%
    * Polynomial Kernel Accuracy: 50.51%
* **Custom CNN (Group) + 5-Fold CV with SVM:** This model, using a 3-layer CNN with 5-fold cross-validation to feed features into SVMs, achieved the highest accuracy.
    * **Average Accuracy (RBF Kernel): 98.07%**
    * Average Accuracy (Polynomial Kernel): 96.27%

---

### 2. `Alzheimer_CNN.ipynb`

This notebook trains a standard 3-layer CNN model using 5-fold stratified cross-validation. This appears to be the baseline CNN used as the feature extractor in the best-performing `SVM_Alzheimer.ipynb` model.

* **Average Accuracy (5-Folds): 96.33%**

---

### 3. `Alzheimer_Grad_CAM.ipynb`

This notebook is for visualization, not accuracy testing. It generates **Grad-CAM (Gradient-weighted Class Activation Mapping)** heatmaps to show which parts of an image the CNN uses to make predictions.

---

### 4. `Alzheimer_Multilayer_Perceptron.ipynb`

This notebook trains a **Multilayer Perceptron (MLP)**, a type of feedforward neural network, using PyTorch for 150 epochs.

* **Test Accuracy (at Epoch 150): 92.97%**

---

### 5. `Alzheimer_prediction_using cnn_tensorflow.ipynb`

This notebook trains a large CNN (likely a pre-trained model like VGG19, given its ~159M parameters) for 50 epochs.

* **Validation Accuracy:** Reached **80.34%** at epoch 43.
