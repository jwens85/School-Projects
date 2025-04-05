import numpy as np

np.random.seed(42)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def mean_squared_error(Y, ŷ):
    m = len(Y)
    return (1/m) * np.sum((Y - ŷ) ** 2)

def mse_derivative(Y, ŷ):
    m = len(Y)
    return (-2/m) * (Y - ŷ)

W1 = np.random.randn(2, 5)
b1 = np.zeros((1, 5))
W2 = np.random.randn(5, 1)
b2 = np.zeros((1, 1))

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

learning_rate = 0.01
epochs = 1000

for epoch in range(epochs):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    ŷ = sigmoid(Z2)

    loss = mean_squared_error(Y, ŷ)

    dL_dŷ = mse_derivative(Y, ŷ)
    dŷ_dZ2 = sigmoid_derivative(ŷ)
    dZ2_dW2 = A1.T
    dL_dW2 = np.dot(dZ2_dW2, dL_dŷ * dŷ_dZ2)
    dL_db2 = np.sum(dL_dŷ * dŷ_dZ2, axis=0, keepdims=True)
    dZ2_dA1 = W2.T
    dA1_dZ1 = sigmoid_derivative(A1)
    dZ1_dW1 = X.T
    dL_dW1 = np.dot(dZ1_dW1, np.dot(dL_dŷ * dŷ_dZ2, dZ2_dA1) * dA1_dZ1)
    dL_db1 = np.sum(np.dot(dL_dŷ * dŷ_dZ2, dZ2_dA1) * dA1_dZ1, axis=0, keepdims=True)

    W2 -= learning_rate * dL_dW2
    b2 -= learning_rate * dL_db2
    W1 -= learning_rate * dL_dW1
    b1 -= learning_rate * dL_db1

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

Z1 = np.dot(X, W1) + b1
A1 = sigmoid(Z1)
Z2 = np.dot(A1, W2) + b2
ŷ = sigmoid(Z2)

print("\nFinal Prediction for Fibonacci Sequence:", ŷ)
