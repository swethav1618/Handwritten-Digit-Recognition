# Handwritten-Digit-Recognition
Implementation of a machine learning model for handwritten digit recognition using the MNIST dataset and Jupyter Notebook.

---
## 📂 Dataset Handling

- Dataset: [MNIST](http://yann.lecun.com/exdb/mnist/)  
- Preprocessing:
  - Images resized to **28x28 grayscale**.
  - Normalized pixel values.
   
## Custom Dataset Class

We implement a dataset class using `torch.utils.data.Dataset`, including:

- **`__init__()`** – Loads the dataset.  
- **`__len__()`** – Returns dataset length.  
- **`__getitem__()`** – Retrieves an image-label pair at a given index
  
 ## DataLoader
 
Data is batched and shuffled using torch.utils.data.DataLoader.

## 🧠 CNN Model Architecture

- Our CNN uses multiple convolutional and pooling layers with ReLU activation.
- Convolutional layers – extract features from input/digit images.
- ReLU – introduces non-linearity.
- Max Pooling – reduces spatial dimensions while preserving important features.
- Fully connected layers – map extracted features to class probabilities.
- Model Summary
   - Conv Layer → ReLU → Max Pool
   - Conv Layer → ReLU → Max Pool
   - Fully Connected Layer (128)
   - Output Layer (10 classes for digits 0–9)

## ⚙️ Training
Loss Function: CrossEntropyLoss
Optimizer: Adam
- Training is performed for several epochs using mini-batches.  
- The model quickly converges and reaches high accuracy.  

## 📊 Results
Final Test Accuracy: ~99% 🎉
Evaluation Metrics: Accuracy
                    Confusion Matrix

## 🔮 Future Work
Experiment with deeper CNN architectures.
Add dropout/regularization.
Deploy as a web app (Flask/Streamlit)
