# Multiple Linear Regression with Graph

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# ✅ Simple dataset (10 samples, 2 features)
X = np.array([
    [1, 2],
    [2, 1],
    [3, 4],
    [4, 3],
    [5, 5],
    [6, 7],
    [7, 6],
    [8, 8],
    [9, 10],
    [10, 9]
])

y = np.array([3, 3, 7, 7, 10, 13, 13, 16, 19, 19])

# ✅ Train model
model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# ✅ 3D Graph
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Original data points
ax.scatter(X[:,0], X[:,1], y)

# Create regression plane
x1_surf, x2_surf = np.meshgrid(
    np.linspace(X[:,0].min(), X[:,0].max(), 10),
    np.linspace(X[:,1].min(), X[:,1].max(), 10)
)

y_surf = model.intercept_ + model.coef_[0]*x1_surf + model.coef_[1]*x2_surf

# Plot plane
ax.plot_surface(x1_surf, x2_surf, y_surf, alpha=0.3)

ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Output y")

plt.title("Multiple Linear Regression (3D Graph)")
plt.show()
