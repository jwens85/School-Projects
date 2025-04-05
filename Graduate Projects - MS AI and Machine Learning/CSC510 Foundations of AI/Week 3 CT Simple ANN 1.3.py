import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Define mean squared error loss function
def mean_squared_error(Y, ŷ):
    m = len(Y)
    return (1/m) * np.sum((Y - ŷ) ** 2)

def mse_derivative(Y, ŷ):
    m = len(Y)
    return (-2/m) * (Y - ŷ)

# Define input (X) and expected output (Y)
X = np.array([
    [0, 1],
    [1, 1],
    [1, 2],
    [2, 3],
    [3, 5],
    [5, 8],
    [8, 13],
    [13, 21]
])
Y = np.array([
    [1],
    [2],
    [3],
    [5],
    [8],
    [13],
    [21],
    [34]
])

# Normalize input and output
max_fib = np.max(Y)
X = X / max_fib
Y = Y / max_fib
#(Grimoire, 2025)
# Initialize weights and biases
W1 = np.random.randn(2, 5)
b1 = np.zeros((1, 5))
W2 = np.random.randn(5, 1)
b2 = np.zeros((1, 1))

# Hyperparameters
learning_rate = 0.1
epochs = 1000000

# Training loop
predictions_over_time = {}  # Dictionary to store predictions at each checkpoint

for epoch in range(epochs):
    # Forward pass
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)  # Sigmoid in hidden layer
    Z2 = np.dot(A1, W2) + b2
    ŷ = Z2  # Linear activation in output layer

    # Compute loss
    loss = mean_squared_error(Y, ŷ)

    # Store predictions at every 100 epochs
    if epoch % 100000 == 0:
        predictions_over_time[epoch] = ŷ[-1][0] * max_fib  # Store last prediction
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

    # Backpropagation
    dL_dŷ = mse_derivative(Y, ŷ)
    dŷ_dZ2 = 1  # Derivative of linear function is 1
    dZ2_dW2 = A1.T
    dL_dW2 = np.dot(dZ2_dW2, dL_dŷ * dŷ_dZ2)
    dL_db2 = np.sum(dL_dŷ * dŷ_dZ2, axis=0, keepdims=True)
    dZ2_dA1 = W2.T
    dA1_dZ1 = sigmoid_derivative(A1)
    dZ1_dW1 = X.T
    dL_dW1 = np.dot(dZ1_dW1, np.dot(dL_dŷ * dŷ_dZ2, dZ2_dA1) * dA1_dZ1)
    dL_db1 = np.sum(np.dot(dL_dŷ * dŷ_dZ2, dZ2_dA1) * dA1_dZ1, axis=0, keepdims=True)
#(Grimoire, 2025)
    # Gradient descent update
    W2 -= learning_rate * dL_dW2
    b2 -= learning_rate * dL_db2
    W1 -= learning_rate * dL_dW1
    b1 -= learning_rate * dL_db1
#(Grimoire, 2025)
# Print final predictions separately
print("\nFinal Predictions for Fibonacci Sequence:")
for epoch, prediction in predictions_over_time.items():
    print(f"Epoch {epoch}, Prediction: {prediction:.6f}")

# Print final prediction for epoch 1000000
print(f"\nFinal Prediction (Epoch {epochs}): {predictions_over_time[epochs - 100000]:.6f}")
