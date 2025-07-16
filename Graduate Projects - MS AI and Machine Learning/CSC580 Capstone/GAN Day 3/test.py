import numpy as np
import keras
from keras.datasets import cifar10

# Load CIFAR-10
(X, y), (_, _) = cifar10.load_data()

# Filter class 8 (horses)
X = X[y.flatten() == 8]

# Normalize to range [-1, 1]
X = (X / 127.5) - 1.0

print("Filtered shape:", X.shape)
print("Min pixel value:", X.min(), "Max pixel value:", X.max())
