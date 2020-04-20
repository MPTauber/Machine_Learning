## Reimplement the simple linear regression case study of Section 15.4 using the average yearly temperature data. 
## How does the temperature trend compare to the average January high temperatures?

import pandas as pd

nyc = pd.read_csv("ave_yearly_temp_nyc_1895-2017 (2).csv")

nyc.columns = ["Date","Temperature","Anomaly"]

nyc.Date = nyc.Date.floordiv(100) # used for integer division of the dataframe with 100

print(nyc.head(3))

### SPLITTING THE DATA FOR TRAINING AND TESTING
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    nyc.Date.values.reshape(-1,1), nyc.Temperature.values, # reshape infers the number of rows (124) with -1, to fit into 1 column
    random_state=11
)

## TRAINING THE MODEL
from sklearn.linear_model import LinearRegression

linear_regression= LinearRegression()

linear_regression.fit(X=x_train, y=y_train)

LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False) # fit returns the estimator

## LinearRegression tries to ifnd the best fitting regression line by iteratively adjusting the slope and intercept values
# to minimize the sum of the squares of the data points' distances from the line.

print(linear_regression.coef_)
print(linear_regression.intercept_)
# y =mx+b
# So:
# y = 0.03157x - 7.8929
 
### TESTING THE MODEL
predicted = linear_regression.predict(x_test)
expected = y_test

for p,e in zip(predicted[::5], expected[::5]):
    print(f"predicted: {p:.2f}, expected: {e:.2f}")
    ## seems not too far off

### PREDICTING FUTURE TEMPERATURES AND ESTIMATING PAST TEMPERATURES
predict = (lambda x: linear_regression.coef_ * x +
linear_regression.intercept_)

print(predict(2019)) # 55.85584

print(predict(1890)) # 51.7827

### VISUALIZING THE DATASET WITH THE REGRESSION LINE
