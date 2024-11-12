# Dog vs Cat Classification Model

This repository contains a machine-learning model for classifying images of dogs and cats. The model uses a Convolutional Neural Network (CNN) architecture to distinguish between images of dogs and cats. The project includes data preprocessing, model building, training, evaluation, and prediction steps.

## Overview

This project aims to build a classification model that can accurately distinguish between images of dogs and cats. The dataset is pre-processed to ensure that the images are resized and normalized for training the model. We use a deep learning approach with a Convolutional Neural Network (CNN) built using TensorFlow/Keras to achieve high accuracy in classification.

## Technologies Used

- **Python**: Programming language used for model building.
- **TensorFlow/Keras**: Deep learning framework for building the CNN model.
- **NumPy**: For data manipulation and numerical operations.
- **Matplotlib**: For data visualization and plotting.
- **OpenCV**: For image processing.
- **Pandas**: For handling data frames and CSV files.

### Dataset Structure:
- **Training Set**: Images labelled 'dog' or 'cat'.
- **Test Set**: Images to test the model’s predictions.

## Model

The model used in this project is a Convolutional Neural Network (CNN), which is well-suited for image classification tasks. The model architecture consists of multiple convolutional layers, max-pooling layers, and fully connected layers for the classification output.

### Model Architecture:
- **Conv2D** layers for extracting features from the images.
- **MaxPooling2D** layers to reduce spatial dimensions.
- **Dense** layers are used for classification, with a final softmax layer for binary classification.

## Training

The model is trained using the training dataset, with image augmentation applied to prevent overfitting. The training process includes the following steps:
1. Data Preprocessing (Resizing and Normalization)
2. Model Compilation using Adam optimizer.
3. Model Training with a batch size of 32 and epochs set to 10.
4. Saving the trained model for future inference.

