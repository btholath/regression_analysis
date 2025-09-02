Great one 👌 — let’s unpack **Relative Absolute Error (RAE)**, because it’s a metric that’s often less obvious than RMSE or R².

---

## 🎯 What is Relative Absolute Error (RAE)?

* It measures **how well your regression model predicts** compared to a **naïve baseline model** (like always predicting the mean of `y`).
* It looks at the **absolute errors** (ignoring signs), then compares the model’s total error to the baseline’s total error.

---

## 🧮 Formula

$$
\text{RAE} = \frac{\sum_{i=1}^n |y_i - \hat{y}_i|}{\sum_{i=1}^n |y_i - \bar{y}|}
$$

Where:

* $y_i$ = actual values
* $\hat{y}_i$ = predicted values
* $\bar{y}$ = mean of actual values

---

## 📊 How to Interpret

* **RAE = 0** → perfect model (no error).
* **RAE < 1** → model is better than predicting the mean.
* **RAE = 1** → model is *just as good as guessing the average*.
* **RAE > 1** → model is worse than guessing the average!

---

## 🐍 Python Example (with your regression setup)

```python
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
```

---

## ✅ Why is RAE useful?

* Unlike RMSE, it has an **easy-to-understand scale**:

  * Less than 1 → good
  * Equal to 1 → meh
  * Greater than 1 → bad
* Helps you know if your model actually beats a “dumb baseline” predictor.

---