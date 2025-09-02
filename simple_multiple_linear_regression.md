
---

# 📊 Simple vs. Multiple Linear Regression

| Aspect             | **Simple Linear Regression (SLR)**                                        | **Multiple Linear Regression (MLR)**                                                       |
| ------------------ | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| **Definition**     | Predicts a dependent variable (Y) using **one** independent variable (X). | Predicts a dependent variable (Y) using **two or more** independent variables (X₁, X₂, …). |
| **Formula**        | $Y = β₀ + β₁X + ε$                                                        | $Y = β₀ + β₁X₁ + β₂X₂ + … + βₙXₙ + ε$                                                      |
| **Use Case**       | Relationship between hours studied and exam marks.                        | Relationship between exam marks and hours studied, attendance, and sleep.                  |
| **Visualization**  | A straight line on a 2D scatter plot.                                     | A plane (for 2 variables) or hyperplane (for 3+) in higher dimensions.                     |
| **Complexity**     | Easier to interpret.                                                      | More complex, risk of multicollinearity.                                                   |
| **Python Example** | `LinearRegression()` with one feature (X).                                | `LinearRegression()` with multiple features (X₁, X₂, …).                                   |

---

## 🧮 Example 1: Simple Linear Regression (SLR)

Dataset: `02_students.csv`

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("./simple_linear_regression/02_students.csv")

# Independent variable: Hours
X = df[["Hours"]]   # 2D
y = df["Marks"]

# Train model
slr = LinearRegression()
slr.fit(X, y)

print("SLR Equation: Marks = {:.2f} + {:.2f} * Hours".format(slr.intercept_, slr.coef_[0]))

# Plot
plt.scatter(X, y, color="blue", label="Data")
plt.plot(X, slr.predict(X), color="red", label="SLR Fit")
plt.xlabel("Hours")
plt.ylabel("Marks")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()
```

👉 This draws a **line** that predicts Marks from Hours.

---

## 🧮 Example 2: Multiple Linear Regression (MLR)

Let’s extend the dataset: Suppose we add **Attendance** and **Sleep Hours**.

```python
import pandas as pd
from sklearn.linear_model import LinearRegression

# Example dataset
data = {
    "Hours": [5, 6, 8, 9, 10],
    "Attendance": [80, 85, 90, 92, 95],
    "Sleep": [6, 7, 6, 5, 7],
    "Marks": [65, 70, 78, 82, 85]
}
df = pd.DataFrame(data)

# Independent variables: Hours, Attendance, Sleep
X = df[["Hours", "Attendance", "Sleep"]]
y = df["Marks"]

# Train model
mlr = LinearRegression()
mlr.fit(X, y)

print("MLR Intercept (β₀):", mlr.intercept_)
print("MLR Coefficients (β):", mlr.coef_)
print("Equation: Marks = {:.2f} + {:.2f}*Hours + {:.2f}*Attendance + {:.2f}*Sleep"
      .format(mlr.intercept_, mlr.coef_[0], mlr.coef_[1], mlr.coef_[2]))
```

👉 This fits a **plane (or hyperplane)** that predicts Marks from **Hours, Attendance, and Sleep**.

---

## 🎯 Key Takeaways for You

* **SLR** = one factor → easy to draw as a line.
* **MLR** = many factors → better predictions but harder to visualize.
* Both use the **same idea**: find β-values that minimize error (OLS).
* In real life:

  * **SLR** → “Does studying more hours improve my grades?”
  * **MLR** → “Grades depend on studying + attending classes + sleeping well.”

---

👉 Would you like me to also **plot the MLR case in 3D** (Hours vs Attendance vs Marks) so you can *see* the fitted plane next to the SLR line?
