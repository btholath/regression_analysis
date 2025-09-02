"""

"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# Load dataset
df = pd.read_csv("./simple_linear_regression/02_students.csv")
X = df[["Hours"]].values
y = df["Marks"].values

# Split train/test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

# Train model
model = LinearRegression()
model.fit(x_train, y_train)

# Predictions
y_pred = model.predict(x_test)

# --- Relative Absolute Error ---
num = np.sum(np.abs(y_test - y_pred))
den = np.sum(np.abs(y_test - np.mean(y_test)))
rae = num / den

print("Relative Absolute Error (RAE):", rae)
