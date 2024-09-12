# SignBridge-
A cutting-edge Sign Language Detection System employing Machine Learning, Computer Vision, and Deep Learning techniques to translate sign language gestures into text in real-time, enhancing communication accessibility for the deaf and hard-of-hearing community.

## Overview
This project aims to build a Convolutional Neural Network (CNN) to recognize American Sign Language (ASL) from images. The model is trained on a dataset of 86,972 images and validated on a test set of 55 images, each labeled with the corresponding sign language letter or action.

## Dataset
The dataset used in this project is sourced from Kaggle and contains images for each letter of the ASL alphabet. The training and testing images are organized in separate directories, with the training images further sorted into subdirectories by label.

## Model Architecture
The model architecture is Sequential and utilizes various layers:

- Conv2D layers for feature extraction
- MaxPooling2D layers for spatial dimension reduction
- Flatten layer to convert 3D feature maps to 1D vectors
- Dense layers for classification
- BatchNormalization for normalization and regularization
- LeakyReLU and ReLU activations for non-linearity
- Softmax activation in the final layer for class probabilities
  
## Data Preparation

### Loading and Preprocessing the Dataset

The dataset is organized into folders, each representing a class label for the ASL alphabet (e.g., "A" for the letter A).

- Images are resized to a uniform size of 64x64 pixels .
- Pixel values are normalized to the range [0, 1] for faster convergence during training.
- Labels are encoded using a dictionary to map class labels to unique integers.

### Reshaping the Input Data

Input data (x_train, x_test) is reshaped to fit the model's expected input shape, including the color channels.

### One-hot Encoding the Labels

Labels (y_train, y_test) are one-hot encoded for compatibility with categorical crossentropy loss.

## Compilation and Training

The model is compiled with the Adam optimizer and categorical crossentropy loss. Accuracy is used as the evaluation metric.

- Training is performed using the specified training data with validation data for performance monitoring.

## Saving and Loading the Model

After training, the model is saved to 'my_model.h5' for future use. The saved model can be loaded for predictions on new images.

## Making Predictions

To make predictions on new images:

- Load the image
- Resize to the model's input size (64x64 pixels)
- Normalize the pixel values
- Pass the image to the model
- Obtain a probability distribution over classes
- Select the class with the highest probability as the predicted class

## Real-time Prediction with OpenCV

The real-time prediction script uses a webcam feed:

- Captures frames in real time
- Preprocesses each frame
- Uses the trained model for ASL sign prediction
- Displays predicted labels on the frames in a window for real-time sign language recognition.




