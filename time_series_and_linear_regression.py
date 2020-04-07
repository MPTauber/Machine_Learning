import pandas as pd 

### LOADING THE AVERAGE HIGH TEMPERATURES INTO A DATAFRAME
nyc = pd.read_csv("ave_hi_nyc_jan_1895-2018.csv")

print(nyc.head(3))

### SPLITTING THE DATA FOR TRAINING AND TESTING
# By default this estimator uses all the numeric features in a dataset,
# performing a multiple linear regression.
# We want to perform a simple linear regression using just one feature (column/attribute)
# as an indepndent variable. We will use the feature - Date

# A column in DataFrame is a one-dimensional Series
# Scikit-learn estimators require training and testing dat to be two-dimensional
# Transform Series of n elements into two dimensions containing n rows and one column.

print(nyc.Date.values) #returns NumPy array containing Date column's values

# reshape (-1,1) tells reshape to infer the number of rows, based on the number
# of columns (1) and the number of elements (124) in the array
