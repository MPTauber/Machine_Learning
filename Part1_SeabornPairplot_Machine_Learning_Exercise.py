## Create a Seaborn pairplot graph (the book has an example in Unsupervised Machine Learning for the Iris Datset)
## for the California Housing dataset. Try the matplotlib features for panning and zooming the diagram.
## These are accessible via the icons in the matplotlib window.

## Pairplot is a grid of graphs plotting each feature against itself and the other specified features.

from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 

california = fetch_california_housing() # bunch objects
california_df = pd.DataFrame(california.data, columns=california.feature_names) # california housing data into dataframe
california_df["MedHouseValue"] = pd.Series(california.target) # add prices to the houses

sns.set(font_scale=1.1)
sns.set_style("whitegrid")

grid = sns.pairplot(data=california_df,vars=california_df.columns[0:8])

#plt.show()

