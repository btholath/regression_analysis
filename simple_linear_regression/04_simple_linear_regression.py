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

"""
Linear Regression is about finding a straight line

You have two things that might be related.

Example here: Hours of study (independent variable) and Marks (dependent variable).

We ask: If I study more, how does that affect my marks?

Equation of a Line

Remember from math:

𝑦=𝑚𝑥+𝑏
y=mx+b

In regression we call it:

Marks=𝛽0+𝛽1×Hours
Marks=β0+β1×Hours
β₀ (b, intercept) → where the line starts (when Hours = 0).
β₁ (m, slope) → how much Marks change for each extra Hour studied.

Dots vs. Line
The blue dots are your actual data (real students’ Hours and Marks).
The red line is the computer’s “best guess” line through those dots.

The goal is: find the line that best fits the pattern in the dots.

Prediction
Once you have the line, you can predict marks for any hours studied.
Example: If the line says

Marks=30+5×Hours
Marks=30+5×Hours

and a student studies 6 hours:

Marks=30+5×6=60
Marks=30+5×6=60
🧠 What You’re Really Learning

Patterns: Data often follows a trend, and math helps us describe it.

Modeling: We build models (like lines) to explain or predict the real world.

Critical thinking: Not every point lies exactly on the line — life is messy! The line is our best guess.
"""