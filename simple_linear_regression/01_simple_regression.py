# --------------------------------------------------------------
# Simple Linear Regression
# Predict the marks obtained by a student based on hours of study
# --------------------------------------------------------------


import sys, os
import sys
sys.path.append('/workspaces/regression_analysis')


# Import Pandas for data processing
import pandas as pd



# Read the CSV file
dataset = pd.read_csv('./simple_linear_regression/01_students.csv')
df = dataset.copy()

# Split into X (Independent) and Y (predicted)
X = df.iloc[:, :-1]
Y = df.iloc[:,  -1]


# Create the Training and Test datasets
"""
X → your input features (e.g., hours studied).
Y → your target labels (e.g., marks).

test_size=0.3 → 30% of the dataset will be set aside for testing, 70% for training.
random_state=1234 → fixes the random shuffle, so results are reproducible.

Returns → (x_train, x_test, y_train, y_test).
"""
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =     \
train_test_split (X, Y, test_size = 0.3, random_state=1234)

# Train the Simple Linear Regression
from sklearn.linear_model import LinearRegression
std_reg = LinearRegression()
std_reg.fit(x_train, y_train)

# Predict the results
y_predict = std_reg.predict(x_test)

# Get the R-Squared 
slr_score = std_reg.score(x_test, y_test)

# Coefficient and Intercept
slr_coefficient = std_reg.coef_
slr_intercept = std_reg.intercept_

# Equation of the line
#   y = 34.27 + 5.02 * X

# Calculate the errors using RMSE 
from sklearn.metrics import mean_squared_error
import math

slr_rmse = math.sqrt(mean_squared_error(y_test, y_predict))

# plot the result using matplotlib 
import matplotlib.pyplot as plt

plt.scatter(x_test, y_test)
plt.plot(x_test, y_predict)
plt.ylim(ymin=0)
plt.show()
plt.savefig("simple_linear_regression_residuals.png")
















