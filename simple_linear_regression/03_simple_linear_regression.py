# --------------------------------------------------------------
# Simple Linear Regression on 02_students.csv
# Predict Marks from Hours of Study
# --------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# --- Load dataset ---
df = pd.read_csv("./simple_linear_regression/02_students.csv")

# Independent variable (Hours) and target (Marks)
X = df[["Hours"]].values   # keep 2D for sklearn
y = df["Marks"].values     # 1D

# --- Train model ---
model = LinearRegression()
model.fit(X, y)

# --- Parameters ---
print("Slope (coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)
print(f"Equation: y = {model.intercept_:.2f} + {model.coef_[0]:.2f} * Hours")

# --- Predict ---
y_pred = model.predict(X)

# --- Plot ---
plt.figure(figsize=(7, 5))
plt.scatter(X, y, color="blue", label="Actual data")
plt.plot(X, y_pred, color="red", label="Regression line")
plt.xlabel("Hours of Study")
plt.ylabel("Marks Obtained")
plt.title("Simple Linear Regression - Student Marks vs Hours")
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig("/workspaces/regression_analysis/plots/simple_linear_regression_fit_predict_marks_from_hours_of_study.png")
