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
X = df[["Hours"]].values   # 2D for sklearn
y = df["Marks"].values     # 1D

# --- Train model ---
model = LinearRegression()
model.fit(X, y)

# --- Parameters ---
slope = model.coef_[0]
intercept = model.intercept_
print(f"β₀ (Intercept): {intercept:.2f}")
print(f"β₁ (Slope): {slope:.2f}")
print(f"Equation: Marks = {intercept:.2f} + {slope:.2f} * Hours")

# --- Predict ---
y_pred = model.predict(X)

# --- Plot regression line ---
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color="blue", label="Actual data")
plt.plot(X, y_pred, color="red", label="Regression line")

# --- Mark β₀ (intercept) ---
plt.scatter(0, intercept, color="green", s=100, marker="o", label=f"β₀ = {intercept:.2f}")
plt.axhline(y=intercept, color="green", linestyle="--", linewidth=1)

# --- Show dotted lines for one example ---
example_hour = 6
example_pred = model.predict(np.array([[example_hour]]))[0]

# Vertical dotted line (Hours → Predicted Marks)
plt.axvline(x=example_hour, ymin=0, ymax=1, color="gray", linestyle="--", linewidth=1)
# Horizontal dotted line (Predicted Marks level)
plt.axhline(y=example_pred, xmin=0, xmax=1, color="gray", linestyle="--", linewidth=1)

# Highlight the predicted point
plt.scatter(example_hour, example_pred, color="purple", s=100, marker="x",
            label=f"Prediction at Hours={example_hour}")

# --- Labels and legend ---
plt.xlabel("Hours of Study")
plt.ylabel("Marks Obtained")
plt.title("Simple Linear Regression - Student Marks vs Hours")
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig("/workspaces/regression_analysis/plots/04_simple_linear_regression_fit_predict_marks_from_hours_of_study.png")
