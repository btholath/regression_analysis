import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

"""
When we build a machine learning model (like linear regression), we want to measure how well it generalizes to new, unseen data — 
not just how well it memorizes the training data.

Training set → used to “teach” the model (fit the regression line).
    Test set → used to evaluate the trained model’s performance on unseen data.

Without this split, you might get overfitting: the model looks perfect on training data but fails on new examples.
"""

# Sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 3, 2, 3, 5])

# Train model
model = LinearRegression()
model.fit(X, y)

print("Slope:", model.coef_)
print("Intercept:", model.intercept_)

# Predict
y_pred = model.predict(X)

# Plot
plt.scatter(X, y, color="blue")
plt.plot(X, y_pred, color="red")
plt.show()
