# Simple Decision Tree Example

import numpy as np
from sklearn.tree import DecisionTreeClassifier

# ✅ Small easy dataset
# Example: [Study Hours, Sleep Hours]
X = np.array([
    [1, 5],
    [2, 6],
    [3, 7],
    [6, 3],
    [7, 2],
    [8, 1]
])

# Labels: 0 = Fail, 1 = Pass
y = np.array([0,0,0,1,1,1])

# ✅ Create Decision Tree model
model = DecisionTreeClassifier()

# ✅ Train model
model.fit(X, y)

# ✅ Predict new data
prediction = model.predict([[5,4]])
print("Prediction:", prediction)
