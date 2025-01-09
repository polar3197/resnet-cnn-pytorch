# ResNet-based CNN in PyTorch

## Overview
This CIFAR10 classification project, assigned by CS 189 at UC Berkeley, implements a convolutional neural network (CNN) using PyTorch. I designed my model after the one described in the research paper 'Deep Residual Learning for Image Recognition' (https://doi.org/10.48550/arXiv.1512.03385).

## Dataset
CIFAR10

## Implementation
- Framework: PyTorch
- Model: ResNet-based CNN with simple three res-block structure
- Features:
    - Custom ResNet-inspired architecture with three residual blocks and skip connections.
    - Xavier initialization applied to weights for improved convergence.
    - Dropout regularization (p=0.6) before the fully connected layer to reduce overfitting.
    - StepLR learning rate scheduler with decay factor of 0.1 every 10 epochs.
    - CrossEntropyLoss for multi-class classification.

## Results
Test accuracy 82.02%


## Setup
1. Create a virtual environment:
    ```sh
    python3 -m venv venv
    source venv/bin/activate  # On macOS/Linux
    venv\Scripts\activate     # On Windows
    pip install -r requirements.txt
    ```
