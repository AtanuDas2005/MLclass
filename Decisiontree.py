import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------
# 1) Dataset using arrays
# -------------------------------
X = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2],
    [5.0, 3.4, 1.5, 0.2],
    [4.6, 3.1, 1.5, 0.2],
    [5.4, 3.9, 1.7, 0.4],

    [6.4, 3.2, 4.5, 1.5],
    [6.9, 3.1, 4.9, 1.5],
    [5.5, 2.3, 4.0, 1.3],
    [6.5, 2.8, 4.6, 1.5],
    [5.7, 2.8, 4.5, 1.3],

    [6.5, 3.0, 5.2, 2.0],
    [6.2, 3.4, 5.4, 2.3],
    [5.9, 3.0, 5.1, 1.8],
    [6.3, 2.9, 5.6, 1.8],
    [7.1, 3.0, 5.9, 2.1]
])

# Labels: 0=setosa, 1=versicolor, 2=virginica
y = np.array([
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2
])

# -------------------------------
# 2) Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -------------------------------
# 3) Train Decision Tree
# -------------------------------
model = DecisionTreeClassifier(criterion="gini", random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# 4) Predictions
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# 5) Accuracy + Report
# -------------------------------
acc = accuracy_score(y_test, y_pred)
print("✅ Accuracy:", acc * 100, "%")

print("\n✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# 6) Tree Diagram (Plot)
# -------------------------------
plt.figure(figsize=(14, 7))
plot_tree(
    model,
    filled=True,
    feature_names=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    class_names=["setosa", "versicolor", "virginica"]
)
plt.title("Decision Tree Diagram")
plt.show()
