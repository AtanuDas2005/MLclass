import numpy as np
import matplotlib.pyplot as plt

# Small easy dataset (10 values)
X = np.array([1,2,3,4,5,6,7,8,9,10], dtype=float)
y = np.array([2,4,5,4,5,7,8,9,10,12], dtype=float)

# Initialize parameters
m = 0      # slope
c = 0      # intercept
lr = 0.01  # learning rate
epochs = 1000

n = len(X)

# Gradient Descent Loop
for i in range(epochs):

    y_pred = m*X + c

    # gradients
    dm = (-2/n) * np.sum(X * (y - y_pred))
    dc = (-2/n) * np.sum(y - y_pred)

    # update parameters
    m = m - lr * dm
    c = c - lr * dc

print("Slope:", m)
print("Intercept:", c)

# Plot result
plt.scatter(X, y)
plt.plot(X, m*X + c)
plt.title("Gradient Descent Linear Regression")
plt.show()
