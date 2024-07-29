import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 5 + 2 * X + np.random.randn(100, 1) * 1.5  # Modified data

# Add x0 = 1 to each instance (bias term)
X_b = np.c_[np.ones((100, 1)), X]

# Gradient Descent
def gradient_descent(X, y, lr=0.01, n_iterations=1000):
    m = len(X)
    theta = np.random.randn(2, 1)
    for iteration in range(n_iterations):
        gradients = 2/m * X.T.dot(X.dot(theta) - y)
        theta -= lr * gradients
    return theta

# Run Gradient Descent
theta = gradient_descent(X_b, y)

# Plot results
plt.plot(X, y, 'b.')
plt.plot(X, X_b.dot(theta), 'r-')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Gradient Descent with Modified Data')
plt.show()

print("Theta:", theta)
