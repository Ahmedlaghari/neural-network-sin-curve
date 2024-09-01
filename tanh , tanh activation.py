
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0, 2 * np.pi, 1000).reshape(1, -1)  # 1000 points between 0 and 2π
y = np.sin(x)  # Corresponding sine values

# Define the neural network structure
n_x = x.shape[0]  # Input layer size (1 feature)
n_h = 10          # Hidden layer size (10 neurons)
n_y = y.shape[0]  # Output layer size (1 output)
m = x.shape[1]  



learning_rate=0.07
n_x =x.shape[0]
n_y=y.shape[0]
n_h=10
m =x.shape[1]

W1 = np.random.randn(n_h,n_x)*0.01
b1 = np.zeros((n_h,1))
W2 = np.random.randn(n_y,n_h)*0.01
b2 = np.zeros((n_y,1))
def tanh_derivative(x):
    return 1 - np.tanh(x)**2
for p in range (30000):
    Z1 = np.dot(W1,x)+b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1)+b2
    A2 = np.tanh(Z2)
    
    
    cost = np.mean((A2 - y)**2)
    dloss= 2 * (A2 - y) / m
    dw21= (tanh_derivative(Z2)*dloss)
    dw2 = np.dot(dw21,A1.T)
    db2 =np.sum(dw21,axis=1,keepdims=True)
    dw11 = np.dot(W2.T, dw21) * tanh_derivative(Z1)
    dw1 =np.dot(dw11,x.T)
    db1 =np.sum(dw11,axis=1,keepdims=True)
    W1 = W1 - learning_rate * dw1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dw2
    b2 = b2 - learning_rate * db2 
    
    if p % 1000 == 0:
        print(f"Iteration {p}, Cost: {cost}")
    
    
    
    
    

Z1 = np.dot(W1,x)+b1
A1 = np.tanh(Z1)
Z2 = np.dot(W2,A1)+b2
A2 = np.tanh(Z2)
predictions = A2

# Plot the results
plt.plot(x.flatten(), y.flatten(), label='True sin(x)', color='blue')
plt.plot(x.flatten(), predictions.flatten(), label='Predicted sin(x)', color='red')
plt.legend()
plt.show()

Z1 = np.dot(W1,1.04)+b1
A1 = np.tanh(Z1)
Z2 = np.dot(W2,A1)+b2
A2 = np.tanh(Z2)
print(A2)