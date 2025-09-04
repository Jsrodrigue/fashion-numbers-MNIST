# CNN on MNIST and Fashion-MNIST

This repository contains a PyTorch implementation of a **two-layer Convolutional Neural Network (CNN)** inspired by the **TinyVGG architecture**, a simplified version of the VGG family of models. The CNN is designed for classification tasks on two widely used benchmark datasets: **MNIST** and **Fashion-MNIST**.

---

## Datasets

- **MNIST**: 70,000 grayscale images of handwritten digits (0–9), 28x28 pixels. The task is to classify each image into the correct digit class.  
- **Fashion-MNIST**: 70,000 grayscale images of clothing items (28x28 pixels) including shoes, shirts, bags, etc. This dataset is more challenging due to visually similar classes like T-shirts vs shirts.

---

## Model Architecture

- **TinyVGG-style CNN** with two convolutional layers and two fully connected layers.
- Each convolutional layer is followed by **ReLU activation** and **MaxPooling**.
- The architecture is simple, compact, and computationally efficient, making it suitable for experimentation and educational purposes.
- Final models in this project use **two hidden layers with 32 units each**, but the code allows experimentation with more layers or units.

---

## Features

- **PyTorch & torchvision** for dataset handling, model definition, and training.
- **Weights & Biases (wandb)** integration:
  - Track training and validation metrics (accuracy, loss).
  - Log hyperparameters (epochs, learning rate, batch size, optimizer, hidden units, dropout).
  - Visualize experiment progress in real time.

- Functions to:
  - Plot misclassified images.
  - Plot confusion matrices.
  - Save and load model state dictionaries.

---

## Results

- **MNIST model**: Test Accuracy **99.42%**, Test Loss **0.02434**  
- **Fashion-MNIST model**: Test Accuracy **92.64%**, Test Loss **0.22318**  

Observations:
- MNIST is easier to classify due to distinct visual patterns.
- Fashion-MNIST is more challenging because of visually similar classes (e.g., T-shirts vs shirts).
- Performance can be improved by experimenting with **more layers, more units, different optimizers, and hyperparameter tuning using wandb**.

---

## Usage

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <repo-folder>

