# Handwritten Character Recognition using a Convolutional Neural Network (CNN)

This project implements a Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset. The model is built using PyTorch and demonstrates a complete machine learning workflow from data preprocessing and model training to evaluation and inference on custom images.

## Table of Contents
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Testing with a Custom Image](#testing-with-a-custom-image)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)

## Features

- **Deep Learning Model**: A robust CNN architecture for high-accuracy image classification.
- **Data Augmentation**: Utilizes random rotations, translations, and perspective shifts to improve model generalization.
- **Optimized Training**: Implements learning rate scheduling (`ReduceLROnPlateau`), weight decay, and batch normalization for stable and efficient training.
- **GPU Acceleration**: Supports CUDA for accelerated training and inference.
- **Model Persistence**: Saves the best performing model based on validation loss for later use.
- **Inference Pipeline**: Includes a script to test the model on the MNIST test set or predict digits from custom images.
- **Modular Code**: The CNN model is defined in a separate class (`CNN_Class.py`) for better organization and reusability.

## Model Architecture

The CNN is composed of three convolutional blocks followed by two fully-connected layers:

1.  **Convolutional Block 1**: 64 filters, 3x3 kernel, LeakyReLU activation, Batch Normalization, Max Pooling, and Dropout.
2.  **Convolutional Block 2**: 128 filters, 3x3 kernel, LeakyReLU activation, Batch Normalization, Max Pooling, and Dropout.
3.  **Convolutional Block 3**: 256 filters, 3x3 kernel, LeakyReLU activation, Batch Normalization, Max Pooling, and Dropout.
4.  **Fully-Connected Layer 1**: 512 units with LeakyReLU activation, Batch Normalization, and Dropout.
5.  **Fully-Connected Layer 2 (Output)**: 10 units corresponding to the 10 digit classes (0-9).

## Technologies Used

- **Python 3.x**
- **PyTorch**: Core machine learning framework for building and training the neural network.
- **Torchvision**: For accessing the MNIST dataset and for image transformations.
- **NumPy**: For numerical operations, particularly in image preprocessing.
- **Matplotlib**: For visualizing images and predictions.
- **Pillow (PIL)**: For image loading and manipulation.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Handwritten-Character-Recognition-CNN.git
    cd Handwritten-Character-Recognition-CNN
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install torch torchvision numpy matplotlib pillow
    ```

## Usage

### Training the Model

To train the CNN model, run the `main.py` script. The script will download the MNIST dataset, perform training and validation, and save the best model to `./models/cnn_model.pth`.

```bash
python main.py
```

### Testing with a Custom Image

To test the trained model on a custom image of a handwritten digit, place your image (e.g., `test.png`) in the `./data/test/` directory and run the `test.py` script.

```bash
python test.py
```

The script will preprocess the image, predict the digit, display the preprocessed image with the prediction, and print the prediction probabilities for each class.

## Project Structure

```
.
├── CNN_Class.py              # Defines the CNN model architecture
├── main.py                   # Main script for training the model
├── test.py                   # Script for testing the model on custom images
├── models/
│   └── cnn_model.pth         # Saved model weights
├── data/
│   ├── MNIST/                # MNIST dataset (downloaded automatically)
│   └── test/
│       └── test.png          # Example test image
└── debug/
    └── debug_preprocessed.png # Preprocessed image for debugging
```

## Future Improvements

- **Hyperparameter Tuning**: Implement a systematic search (e.g., Grid Search, Bayesian Optimization) to find the optimal hyperparameters.
- **More Complex Architectures**: Experiment with more advanced architectures like ResNet or InceptionNet for potentially higher accuracy.
- **Expanded Dataset**: Train the model on other character datasets like EMNIST to recognize letters as well as digits.
- **Web Interface**: Create a simple web application (e.g., using Flask or Gradio) to allow users to upload images and get predictions.
