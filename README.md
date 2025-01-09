## **ResNet-based CNN in PyTorch**

### **Overview**
This CIFAR-10 classification project, assigned by **CS 189 at UC Berkeley**, implements a convolutional neural network (CNN) using **PyTorch**. The model design is loosely inspired by the ResNet architecture described in the research paper **[Deep Residual Learning for Image Recognition](https://doi.org/10.48550/arXiv.1512.03385)** by Kaiming He et al.

### **Dataset**
- **Dataset**: CIFAR-10  
  The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 50,000 images for training and 10,000 for testing.

### **Implementation**
- **Framework**: PyTorch  
- **Model**: Custom ResNet-based CNN with a simple three residual-block structure.

#### **Features**:
- **Custom ResNet-inspired architecture**:
  - Three residual blocks with increasing filter sizes (64, 128, and 256).
  - Skip connections and 1x1 convolution downsampling for dimensional consistency.
- **Xavier initialization** applied to weights for improved convergence.
- **Dropout regularization** (`p=0.6`) before the fully connected layer to reduce overfitting.
- **StepLR learning rate scheduler** with a decay factor of 0.1 every 10 epochs.
- **CrossEntropyLoss** for multi-class classification.

### **Results**
- **Test accuracy**: 82.02%  
  The model achieves 82.02% accuracy on the CIFAR-10 test set after training for 10 epochs.

### **Setup**
1. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   pip install -r requirements.txt
   ```

2. **Run the Jupyter notebook** or Python script:
   - To run the notebook:
     ```bash
     jupyter notebook ResNet_CIFAR10.ipynb
     ```
   - To run the script (if exported to `.py`):
     ```bash
     python ResNet_CIFAR10.py
     ```

### **Usage**
- The project can be used to train a ResNet-based CNN on CIFAR-10 or other similar datasets.
- The final trained model can be tested or fine-tuned further for custom image classification tasks.

