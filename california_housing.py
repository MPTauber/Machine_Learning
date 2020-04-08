from sklearn.datasets import fetch_california_housing

california = fetch_california_housing() # bunch objects

print(california.DESCR)

print(california.data.shape) # two dimensional (20640 samples / rows, 8 features / variables)

print(california.target.shape) # one dimensional
# the 8 features from the data object above all point to one 

print(california.feature_names)



# Creating a dataframe for the cali data
import pandas as pd 

pd.set_option("precision", 4) # a 4 digit precision for floats
pd.set_option("max_columns",9) # display up to 9 columns in DataFRame outputs
pd.set_option("display.width", None) # auto-detect the display width for wrapping

# creates the initial DataFrame using the data in california.data and with the
# column names specified based on the features of the sample
california_df = pd.DataFrame(california.data, columns=california.feature_names)

california_df["MedHouseValue"] = pd.Series(california.target)

print(california_df.head())

#The keyword argument frac specifies the fraction of the data to select (0.1 for 10%),
# and the keyword argument random_state enables you to seed the random number generator.
# This allows you to reproduce the same "randomly" selected rows
sample_df = california_df.sample(frac=0.1, random_state = 17)


import matplotlib.pyplot as plt 
import seaborn as sns

sns.set(font_scale=2)
sns.set_style("whitegrid")

for feature in california.feature_names:
    plt.figure(figsize=(8, 4.5)) # 8" by 4.5" figure
    sns.scatterplot(
        data=sample_df,
        x=feature,
        y="MedHouseValue",
        hue="MedHouseValue",
        palette="cool",
        legend= False,
    )
#plt.show()

#### SPLITTING THE DATA FOR TRAINING AND TESTING
