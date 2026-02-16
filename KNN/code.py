# Easy KNN Example

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# ✅ Small simple dataset
X = np.array([[1],[2],[3],[6],[7],[8]])   # input values
y = np.array([0,0,0,1,1,1])               # labels

# ✅ Create KNN model (k = 3)
knn = KNeighborsClassifier(n_neighbors=3)

# ✅ Train model
knn.fit(X, y)

# ✅ Predict new value
print(knn.predict([[5]]))
