import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1) Load the dataset ---------------------------------------------
# Try your original path; fall back to current folder if needed.
# we read a list of students: how many hours they studied and what marks they got.
try:
    df = pd.read_csv("./simple_linear_regression/02_students.csv")
except FileNotFoundError:
    df = pd.read_csv("02_students.csv")

# Keep the features (Hours) and the target (Marks)
X = df[["Hours"]].values   # 2D: shape (n, 1)
y = df["Marks"].values     # 1D: shape (n,)

# 2) Split into Train/Test so we can test fairly -------------------
# 70% for training, 30% for testing; fixed seed for repeatability
# we learn from some students (train) and check if we’re good on unseen students (test).
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=1234
)

# 3) Train a simple straight-line model ----------------------------
# draw the best straight line through the cloud of dots.
# LinearRegression().fit(...) finds β₀ (intercept) and β₁ (slope).
model = LinearRegression()
model.fit(x_train, y_train)

beta1 = model.coef_[0]            # slope
beta0 = model.intercept_          # intercept (β₀)
print(f"Line: Marks ≈ {beta0:.2f} + {beta1:.2f} × Hours")
"""
Example from this data:
β₀ ≈ 49.11 → if you studied 0 hours, the line predicts ~49 marks.
β₁ ≈ 2.78 → each extra hour adds ~2.78 marks on average.
"""

# 4) Predict on the test set ---------------------------------------
# use the line to guess the marks of the test students.
y_pred = model.predict(x_test)

# 5) Compute RAE step-by-step --------------------------------------
# Numerator: sum of absolute errors of our model
# we total up our model’s mistakes and compare to the baseline’s mistakes.
num = np.sum(np.abs(y_test - y_pred))

# Baseline: always guess the mean of the *test* marks
# (guess the average for everyone)
baseline = np.mean(y_test)

# Denominator: sum of absolute errors of the baseline
den = np.sum(np.abs(y_test - baseline))

rae = num / den
print(f"Baseline (mean of test marks): {baseline:.2f}")
print(f"Model absolute-error sum: {num:.2f}")
print(f"Baseline absolute-error sum: {den:.2f}")
print(f"Relative Absolute Error (RAE): {rae:.3f}")

# 6) Make a tiny table so students can SEE the math ----------------
# each row shows “how wrong” our guess was vs. how wrong the average guess was.
table = pd.DataFrame({
    "Hours": x_test.flatten(),
    "Actual Marks": y_test,
    "Predicted Marks": np.round(y_pred, 2),
})
table["|Actual - Pred|"] = np.round(np.abs(table["Actual Marks"] - table["Predicted Marks"]), 2)
table["Baseline (mean)"] = round(baseline, 2)
table["|Actual - Baseline|"] = np.round(np.abs(table["Actual Marks"] - baseline), 2)
print("\nHow RAE is built (row by row):")
print(table.to_string(index=False))

# 7) Visual: scatter, best-fit line, β0, and dotted residuals ------
# the dots are students, the line is our model, the dotted lines show “miss distance”.
# we scatter all points, draw the fit line, label the intercept β₀, and draw dotted residuals from each test dot down/up to the line at that same x.
plt.figure(figsize=(7,5))

# All data points
plt.scatter(df["Hours"], df["Marks"], label="Students", zorder=3)

# Best-fit line (use a smooth x range)
x_line = np.linspace(df["Hours"].min(), df["Hours"].max(), 100).reshape(-1, 1)
y_line = model.predict(x_line)
plt.plot(x_line, y_line, label="Best-fit line", zorder=2)

# Mark the intercept β0 at Hours=0
plt.scatter([0], [beta0], s=80, zorder=4)
plt.text(0, beta0, "  β₀ (intercept)", va="bottom")

# Dotted residuals for TEST points (actual → predicted on the line)
for xh, ya in zip(x_test.flatten(), y_test):
    yp = model.predict([[xh]])[0]
    # draw a dotted line from actual point down/up to the line
    plt.plot([xh, xh], [ya, yp], linestyle=":", linewidth=1)

plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Hours vs. Marks — Line Fit, β₀, and Dotted Residuals")
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig("/workspaces/regression_analysis/plots/02_run.png")

"""
a straight line can predict marks from hours: start at β₀ and go up by β₁ each hour.
residual = how far a dot is from the line (our “miss”).
RAE compares our total “miss” to the “always average” strategy.
RAE < 1 means our line is truly helpful!
"""