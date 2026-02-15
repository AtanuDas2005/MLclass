# Simple Linear Regression Example

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# ✅ Easy inbuilt dataset (10 values)
# X = input feature
# y = output value
X = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
y = np.array([2,4,5,4,5,7,8,9,10,12])

# ✅ Create model
model = LinearRegression()

# ✅ Train model
model.fit(X, y)

# ✅ Predict values
y_pred = model.predict(X)

# ✅ Print slope and intercept
print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)

# ✅ Plot graph
plt.scatter(X, y)          # original points
plt.plot(X, y_pred)        # regression line
plt.xlabel("X values")
plt.ylabel("y values")
plt.title("Simple Linear Regression")
plt.show()
