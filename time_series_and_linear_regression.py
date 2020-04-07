import pandas as pd 

### LOADING THE AVERAGE HIGH TEMPERATURES INTO A DATAFRAME
###
nyc = pd.read_csv("ave_hi_nyc_jan_1895-2018.csv")

print(nyc.head(3))

### SPLITTING THE DATA FOR TRAINING AND TESTING
###
# By default this estimator uses all the numeric features in a dataset,
# performing a multiple linear regression.
# We want to perform a simple linear regression using just one feature (column/attribute)
# as an indepndent variable. We will use the feature - Date

# A column in DataFrame is a one-dimensional Series
# Scikit-learn estimators require training and testing dat to be two-dimensional
from sklearn.model_selection import train_test_split
# Transform Series of n elements into two dimensions containing n rows and one column.

print(nyc.Date.values) #returns NumPy array containing Date column's values

# reshape (-1,1) tells reshape to infer the number of rows, based on the number
# of columns (1) and the number of elements (124) in the array
print(nyc.Date.values.reshape(-1,1))

# Transofrmed array will have 124 rows and one column
# 75% for training
# 25% for testing
X_train, X_test, y_train, y_test = train_test_split(
    nyc.Date.values.reshape(-1,1), nyc.Temperature.values, random_state = 11)

print(X_train.shape)
print(X_test.shape)

### TRAINING THE MODEL
###
# To find the best fitting regression line for the data, the LinearRegression
# estimator iteratibely adjusts the slope and intercept to minimize the sum of
# the squares of the data points' distances from the line.
from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()

# the fit method expects the samples and the targets for training
linear_regression.fit(X=X_train, y=y_train)

# Using the model for slope and intercept we can make predictions
# Slope is the estimator's coeff_attribute (m in the equation)
# Intercept is the estimator's intercept attribute (b in the euqation)
# y = mx + b

print(linear_regression.coef_)
print(linear_regression.intercept_)

# Test the model using the data in X_test and check some of the preditions
predicted = linear_regression.predict(X_test)

expected = y_test

for p, e in zip(predicted[::5], expected[::5]): # check every 5th element
    print(f"predicted: {p:.2f}, expected: {e:.2f}")

### PREDICTING FUTURE TEMPERATURES AND ESTIMATING PAST TEMPERATURES
# Use the coefficient and intercept values to make precitions

# lambda implements y = mx + b
predict = (lambda x: linear_regression.coef_ * x + linear_regression.intercept_)

print(predict(2019))
print(predict(1890))

### VISUALIZING THE DATASET WITH THE REGRESSION LINE
###
# Create a scatterplot with a regression line
# Cooler temperatures shown in darker colors

import seaborn as sns

axes = sns.scatterplot(
    data = nyc,
    x = "Date",
    y = "Temperature",
    hue = "Temperature",
    palette = "winter",
    legend = False,
)

axes.set_ylim(10,70)

### CREATING THE REGRESSION LINE
###
# Create an array containing the min and max date values in nyc.Date. 
# These are the x-coordinates of the regression line's start and end points.
# Using those years we can get the y-coordinates for predicted temperature (using the lambda function)
import numpy as np 

x = np.array([min(nyc.Date.values), max(nyc.Date.values)])
print(x)

y = predict(x)
print(y)

### VISUALIZING THE DATASET WITH THE REGRESSION
###
import matplotlib.pyplot as plt

line = plt.plot(x,y)

plt.show()