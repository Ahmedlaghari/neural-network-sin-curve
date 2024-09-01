import numpy as np
import matplotlib.pyplot as plt

# Generate dataset
x = np.linspace(0, 2 * np.pi, 1000).reshape(1, -1)  # 1000 points between 0 and 2π
y = np.sin(x)  # Corresponding sine values

# Define the neural network structure
n_x = x.shape[0]  # Input layer size (1 feature)
n_h = 10          # Hidden layer size (10 neurons)
n_y = y.shape[0]  # Output layer size (1 output)
m = x.shape[1]    # Number of training examples

# Initialize parameters
np.random.seed(0)  # For reproducibility
W1 = np.random.randn(n_h, n_x) * 0.01
b1 = np.zeros((n_h, 1))
W2 = np.random.randn(n_y, n_h) * 0.01
b2 = np.zeros((n_y, 1))

# Activation function and its derivative
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

# Training parameters
learning_rate = 0.07
num_iterations = 30000

# Training loop
for i in range(num_iterations):
    # Forward propagation
    Z1 = np.dot(W1, x) + b1
    A1 = tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = Z2  # Linear activation (no activation function at the output layer)

    # Compute the cost (mean squared error)
    cost = np.mean((A2 - y)**2)

    # Backward propagation
    dA2 = 2 * (A2 - y) / m  # Derivative of cost with respect to A2
    dZ2 = dA2  # Derivative of A2 with respect to Z2 is 1 because A2 is linear
    dW2 = np.dot(dZ2, A1.T)
    db2 = np.sum(dZ2, axis=1, keepdims=True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * tanh_derivative(Z1)
    dW1 = np.dot(dZ1, x.T)
    db1 = np.sum(dZ1, axis=1, keepdims=True)

    # Update parameters
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    # Print the cost every 1000 iterations
    if i % 1000 == 0:
        print(f"Iteration {i}, Cost: {cost}")

# Make predictions
Z1 = np.dot(W1, x) + b1
A1 = tanh(Z1)
Z2 = np.dot(W2, A1) + b2
predictions = Z2

# Plot the results
plt.plot(x.flatten(), y.flatten(), label='True sin(x)', color='blue')
plt.plot(x.flatten(), predictions.flatten(), label='Predicted sin(x)', color='red')
plt.legend()
plt.show()
