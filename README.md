# Handwritten-Digit-Recognition
Implementation of a machine learning model for handwritten digit recognition using the MNIST dataset and Jupyter Notebook.

---
## 📂 Dataset Handling

- Dataset: [MNIST](http://yann.lecun.com/exdb/mnist/)  
- Preprocessing:
  - Images resized to **28x28 grayscale**.
  - Normalized pixel values.
   
## Custom Dataset Class

The custom dataset class acts like a bridge between raw data (MNIST images) and the training loop. It ensures the model gets data in the correct format, one batch at a time, with proper labels.
We implement a dataset class using `torch.utils.data.Dataset`, including:

- **`__init__()`** – Loads the dataset.  
- **`__len__()`** – Returns dataset length.  
- **`__getitem__()`** – Retrieves an image-label pair at a given index
  
Together, these methods make the dataset compatible with **DataLoader**, which is responsible for batching and shuffling during training using torch.utils.data.DataLoader.

## 🧠 CNN Model Architecture, what does Convolution mean in CNNs?
In the context of CNNs, convolution is a mathematical operation where a small filter (also called a kernel) slides over the input image and performs element-wise multiplication and summation. This helps extract local patterns like edges, textures, or shapes.
Think of it like using a magnifying glass to scan different parts of an image—each filter is looking for a specific kind of feature.

- Our CNN uses multiple convolutional and pooling layers with ReLU activation.
- Convolutional layers – extract features from input/digit images.
- ReLU – introduces non-linearity.
- Max Pooling – reduces spatial dimensions while preserving important features.
- Fully connected layers – map extracted features to class probabilities.

## 🧠 Layer-by-Layer Transformation
- conv1 → relu → pool

  Conv1: (1, 28, 28) → (32, 26, 26) (because kernel=3, padding=0, stride=1).

  ReLU: applies elementwise → no shape change.

  MaxPool(2,2): halves H and W → (32, 13, 13).
  
- conv2 → relu → pool
  
  Conv2: (32, 13, 13) → (64, 11, 11) (kernel=3).
  
  ReLU → same shape.

  MaxPool(2,2) → (64, 5, 5).

- Flatten

  (64, 5, 5) → flattened into 64 × 5 × 5 = 1600 features.
  
  So shape becomes (1, 1600).

- fc1 (Linear layer) 1600 → 128.
  
  Shape: (1, 128).

- fc2 (final Linear layer)
  
  128 → 10.

- Shape: (1, 10).

## ⚙️ Training
- Loss Function: CrossEntropyLoss is used to measure classification error.  
- Optimizer: Adam is chosen for efficient weight updates.
- Training is performed for several epochs using mini-batches.  
  
## 📊 Results
- Evaluation Metrics: Accuracy, Confusion Matrix
- Final Test Accuracy: ~99% 🎉
- The model quickly converges and reaches high accuracy.


## 🔮 Future Work
- Experiment with deeper CNN architectures.
- Add dropout/regularization.
- Deploy as a web app (Flask/Streamlit)
