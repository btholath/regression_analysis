# --------------------------------------------------------------
# Simple Linear Regression
# Predict the marks obtained by a student based on hours of study
# --------------------------------------------------------------

from pathlib import Path
import math

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# --- Paths (robust to where you run the script) ---
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "01_students.csv"
PLOTS_DIR = SCRIPT_DIR.parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# --- Load data ---
if not DATA_PATH.exists():
    raise FileNotFoundError(f"CSV not found at: {DATA_PATH}")

df = pd.read_csv(DATA_PATH).copy()

# Expect a single feature column (e.g., Hours) and a target column (e.g., Marks)
# If your CSV has more columns, this still takes "all but last" as X, and last as y
X = df.iloc[:, :-1].values  # ensure 2D
y = df.iloc[:, -1].values   # 1D

# --- Train/test split ---
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1234
)

# --- Train model ---
model = LinearRegression()
model.fit(x_train, y_train)

# --- Predict & metrics ---
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)           # equivalent to model.score(x_test, y_test)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
coef = model.coef_.ravel()              # array of slopes (one for each feature)
intercept = model.intercept_            # intercept

print("=== Simple Linear Regression ===")
print(f"Intercept (b): {intercept:.4f}")
print(f"Coefficients (m): {coef}")
print(f"Equation: y = {intercept:.4f} + {coef[0]:.4f} * X" if coef.size == 1
      else f"Equation: y = {intercept:.4f} + Σ m_i * x_i")
print(f"R^2:  {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# --- Plot: scatter + fitted line (sorted for clean line) ---
# Works best when there's exactly one feature; if multiple, skip drawing the line
plt.figure(figsize=(7, 5))
plt.scatter(x_test[:, 0], y_test, label="Test data")
if X.shape[1] == 1:
    order = np.argsort(x_test[:, 0])
    x_sorted = x_test[order, 0].reshape(-1, 1)
    y_sorted_pred = model.predict(x_sorted)
    plt.plot(x_sorted[:, 0], y_sorted_pred, label="Fitted line")
plt.xlabel("Hours of Study")
plt.ylabel("Marks")
plt.title("Simple Linear Regression: Hours vs Marks")
plt.ylim(bottom=0)
plt.legend()
fit_plot_path = PLOTS_DIR / "simple_linear_regression_fit.png"
plt.tight_layout()
plt.savefig(fit_plot_path)
plt.show()
print(f"Saved plot: {fit_plot_path}")

# --- Optional: residuals plot ---
residuals = y_test - y_pred
plt.figure(figsize=(7, 5))
plt.scatter(y_pred, residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Marks")
plt.ylabel("Residuals (y - ŷ)")
plt.title("Residuals vs Predicted")
resid_plot_path = PLOTS_DIR / "simple_linear_regression_residuals.png"
plt.tight_layout()
plt.savefig(resid_plot_path)
plt.show()
print(f"Saved plot: {resid_plot_path}")
