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

