# Linear Regression Model implementation using NumPy
# Test Code
# Optimization method: Ordinary Least Squares
# Author: Roman Schulze

# import libraries
from linear_regression import *
from sklearn.metrics import r2_score, mean_squared_error as mse
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")


# ----------------------------------------------------------------------------
# 1. Define sample data
# ----------------------------------------------------------------------------

# create an array containing 1000 observations and three columns, each column
# drawn from a normal distribution with mean 10 and standard deviation 3.
X = np.random.normal(10, 3, size=(1000, 3))
# create an array y which slightly deviates from the first column of array X.
# the slight modification is achieved by adding a little bit of noise to each
# observation.
noise = np.random.normal(0, 3, size=(1000, 1))
y = np.transpose(np.array([X[:, 1]])) + noise

# Investigating realtionship between X and y
for i in range(X.shape[1]):
    corr = np.round(np.corrcoef(X[:, i], np.transpose(y))[0, 1], 4)
    print(f"The correlation between X[:, {i}] and y is given by: {corr}.")


# ----------------------------------------------------------------------------
# 2. Use LinearRegressionModel class to model the data from section 1
# ----------------------------------------------------------------------------


print(70 * ("-"))
print("Linear Regression Model using my own class 'LinearRegressonModel'")
print(70 * ("-"))
# Instantiate an object of class LinearRegressionModel
mod = LinearRegressionModel()
# fit the model to the data and store coefficients in coefs
coefs = mod.fit(X, y)
# print the coefficients
print(coefs)
# predict y
y_hat = mod.predict(X)
# check performance using the score method
print(mod.score())
# plot distribution of residuals using the plot method
mod.plot(color="black", alpha=0.7)

# ----------------------------------------------------------------------------
# 3. For comparison run linear regression model using sklearn library
# ----------------------------------------------------------------------------


print(70 * ("-"))
print("Linear Regression Model using sklearn and the same sample data")
print(70 * ("-"))
# train model
reg = LinearRegression().fit(X, y)
print(f"Intercept:{np.round(reg.intercept_, 4)}",
      f"coefficients: {np.round(reg.coef_, 4)}")
# predict y
y_hat = reg.predict(X)
# derive measures of fit
print(f"R_2:{np.round(r2_score(y, y_hat), 4)}")
print(f"RMSE:{np.round(np.sqrt(mse(y, y_hat)), 4)}")
