# Deep Learning Project - Breast Cancer Tissue Classification

This document focuses on the critical task of 'two-stage' breast cancer classification using deep learning techniques applied to high-resolution microscopic images.

## Purpose

The main objective of this project is to develop highly effective models for analyzing medical cell images to achieve two key predictions:
1.  Determine whether a breast tissue sample is **malignant** (cancerous) or **benign** (non-cancerous).
2.  Predict the **specific type of tumor** if cancer is detected.

## Dataset

The project utilizes the **BreaKHis (Breast Cancer Histopathological Database)** from Laboratório Visão Robótica e Imagem.
* **Content:** 7909 high-resolution microscopic images of breast tissue, captured using the SOB method.
* **Tumor Types:**
    * **Benign:** Adenosis (A), Fibroadenoma (F), Phyllodes Tumor (PT), Tubular Adenoma (TA).
    * **Malignant:** Carcinoma (DC), Lobular Carcinoma (LC), Mucinous Carcinoma (MC), Papillary Carcinoma (PC).
* **Variability:** The dataset includes varying numbers of images for each tumor type and different magnifications (40X, 100X, 200X, 400X).

## Methodology

The project was structured into two main stages, employing Convolutional Neural Networks (CNNs) and various image processing techniques.

### Data Organization & Exploration
* Images were organized into stratified folders for training, validation, and testing.
* Initial visual inspection revealed potential "black dots" in malignant tissue images.
* **Texture Analysis:** Gray Level Co-occurrence Matrix (GLCM) was used to extract features and visualize contrast.
* **Color Analysis:** Distribution of RGB color channels was analyzed for both benign/malignant and specific cancer types.

### Preprocessing
* Images were resized to 128x128 pixels to manage RAM and ensure consistency.
* **Data Augmentation:** Techniques like brightness adjustment, rotation and zooming were applied using `ImageDataGenerator` and TensorFlow's Image Processing API to improve model generalization.
* **Normalization & Shuffling:** Images were normalized and shuffled for optimal model training.

### Model Development (Two Stages)

**Stage 1: Binary Classification (Malignant vs. Benign)**
* **Architecture:** Started with a simple Sequential model (two convolutional, two dense layers) and progressively increased complexity.
* **Activation & Loss:** Sigmoid activation for the final layer, binary crossentropy as the loss function.
* **Evaluation:** F1 score was the primary metric due to class imbalance (0.90 achieved), alongside monitoring loss plots for overfitting. Recall was prioritized to minimize false negatives.
* **Optimization:** Keras Hyperband was used for hyperparameter optimization.
* **Experiments:** Explored data augmentation, Macenko Normalization and transfer learning (VGG-16, ResNet-50, Xception), but a custom optimized model performed best.
* **Interpretability:** Grad-CAM visualization confirmed the model focused on the "black dots" initially observed.

**Stage 2: Multiclass Classification (Specific Tumor Type)**
* **Architecture:** Used softmax activation and sparse categorical crossentropy.
* **Initial Performance:** Baseline F1 score of 0,39.
* **Optimization:** Hyperband search with BatchNormalization.
* **Functional API:** Implemented a Functional API with two inputs (images and a CSV file indicating benign/malignant status) to narrow down the prediction task, significantly improving performance.
* **Overfitting Mitigation:** Reduced batch size and incorporated dropout layers.
* **Key Insight:** Discontinuing transfer learning (due to small dataset size and non-medical pre-trained models) significantly improved results, achieving a weighted average F1 score of **0.83**.

## Key Results

* **Stage 1 (Malignant/Benign):** Achieved an F1 score of **0.90** on unseen data. The confusion matrix showed fewer false negatives than false positives, crucial for medical diagnosis.
* **Stage 2 (Specific Tumor Type):** Achieved a weighted average F1 score of **0.83** on unseen data. The model performed well across most classes but struggled with Phyllodes tumor and Papillary carcinoma.

## Conclusion

This project provided deep insights into building Neural Network architectures, applying techniques like transfer learning and Functional API and gaining valuable experience with medical image analysis. It demonstrated how deep neural networks perceive and detect patterns in such images for critical diagnostic purposes.

## Future Work

* Implement stratified k-fold cross-validation for more robust model evaluation.
* Explore pretrained models specifically trained on medical images.
* Investigate additional image transformations.
* Attempt to implement the latest Adopt optimizer for further optimization (pending compatibility issues).
