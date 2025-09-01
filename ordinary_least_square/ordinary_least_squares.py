"""
Awesome 👍 let’s explore **Ordinary Least Squares (OLS)** in a way that makes sense to you.

---

## 🎯 What is OLS?

OLS is the **method computers use to find the best-fitting line** in linear regression.

* We have many blue dots (data points).
* We want a red line that is as close as possible to all the dots.
* **OLS says:** pick the line that minimizes the **sum of the squared vertical distances** (errors) between dots and the line.

---

## 📊 Step-by-Step Example

Imagine this small dataset:

| Hours (X) | Marks (Y) |
| --------- | --------- |
| 1         | 2         |
| 2         | 4         |
| 3         | 5         |
| 4         | 4         |
| 5         | 5         |

---

### 1. Draw a Guess Line

Let’s guess:

$$
y = 1 + 0.8x
$$

---

### 2. Calculate Errors (Residuals)

For each point:

$$
\text{Error} = \text{Actual Y} - \text{Predicted Y}
$$

Example for X=1:

* Predicted Y = 1 + 0.8(1) = 1.8
* Actual Y = 2
* Error = 2 - 1.8 = 0.2

---

### 3. Square the Errors

Squaring makes negatives positive and penalizes big mistakes more.

For X=1: Error² = (0.2)² = 0.04

---

### 4. Add Them All Up

OLS = Sum of all squared errors.
The line with the **smallest OLS value** is the “best fit.”

---

## 🐍 Python Example (OLS with scikit-learn & statsmodels)

```python
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Small dataset
data = {
    "Hours": [1, 2, 3, 4, 5],
    "Marks": [2, 4, 5, 4, 5]
}
df = pd.DataFrame(data)

# Independent variable (with constant for intercept)
X = sm.add_constant(df["Hours"])  # adds β₀ column
y = df["Marks"]

# Fit OLS model
model = sm.OLS(y, X).fit()

# Print summary
print(model.summary())

# Plot
plt.scatter(df["Hours"], y, color="blue", label="Data points")
plt.plot(df["Hours"], model.predict(X), color="red", label="OLS fit line")
plt.xlabel("Hours")
plt.ylabel("Marks")
plt.legend()
plt.show()
```

---

## 🧾 What You’ll See in Output

* **coef** → intercept (β₀) and slope (β₁).
* **R²** → how well the line explains the variation.
* **p-values** → tells if slope is statistically meaningful.

---

✅ **In short:**
OLS is like saying: *“Draw the line where the total squared mistakes are the smallest.”*

---

Would you like me to **reuse your `02_students.csv` file** and run this OLS example on it so you can compare with the scikit-learn linear regression results you already saw?

"""
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Small dataset
data = {
    "Hours": [1, 2, 3, 4, 5],
    "Marks": [2, 4, 5, 4, 5]
}
df = pd.DataFrame(data)

# Independent variable (with constant for intercept)
X = sm.add_constant(df["Hours"])  # adds β₀ column
y = df["Marks"]

# Fit OLS model
model = sm.OLS(y, X).fit()

# Print summary
print(model.summary())

# Plot
plt.scatter(df["Hours"], y, color="blue", label="Data points")
plt.plot(df["Hours"], model.predict(X), color="red", label="OLS fit line")
plt.xlabel("Hours")
plt.ylabel("Marks")
plt.legend()
plt.show()
plt.savefig("/workspaces/regression_analysis/plots/ordinary_least_squares.png")

