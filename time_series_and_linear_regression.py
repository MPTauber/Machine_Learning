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

print(linear_regression.coeff_)
print(linear_regression.intercept_)