## py -3 -m venv ml_venv
## ml_venv/scripts/activate
## pip install sklearn

from sklearn.datasets import load_digits

digits = load_digits()

#print(digits.DESCR)

# a sample is one row
'''
print(digits.data[:2]) # each sample has 64 features (attributes)
# there are 1797 samples
# numbers represent the pixel intensity for each row and column of an image
# tells the intensity of values from 0-16

print(digits.data.shape) # kind of a list object with 64 attributes

print(digits.target[100:120]) # target values of samples
# 100 through 119
# target attribute is the images label (what we're trying to tell the label it's supposed to be)


# Result: [4 0 5 3 6 9 6 1 7 5 4 4 7 2 8 2 2 5 7 9]
# Those are the classes to what the sample belong.
# SO the sample in row 100 belongs to class 4, etc. 

## ---_> this is SUpervised Machine Learning (labeled data)

print(digits.target.shape) # there is only one target, thats why the shape is like that
'''
#####
'''
print(digits.images[13]) # two dimensional array (8x8)
# shows us the pixel intensity of each pixel in the image
# original image is of a blurry 3, that is 8x8 pixels

print(digits.data[13]) # one dimensional (kind of like flattened)
'''
####################
''''
import matplotlib.pyplot as plt
figure, axes = plt.subplots(nrows=4, ncols=6, figsize=(6,4)) # creates a figure objects, and "axes" represents each boxes

# ravel() flattens the image out, so each sample can go into a subplot
for item in zip(axes.ravel(),digits.images, digits.target): # zip allows to bundle any number of objects together (3 here)
    # this way we don't have to use 3 different for loops
    # bundles the axes.ravel(), digits.images, and digits.targets
    axes, image, target = item # this unbundles "item" into the three variables so we can iterate through all of them at the same time
    # cause we're building 24 "pictures"
    axes.imshow(image, cmap = plt.cm.gray_r) #grayscale on each subplot
    axes.set_xticks([]) #removes xticks
    axes.set_yticks([]) #removes y ticks
    axes.set_title(target) # sets title as what the target is supposed to be
plt.tight_layout()

plt.show() # shows the images with targets on top
'''

################### 
# NEXT: Split data for training and testing purpose
# into training set and testing set
from sklearn.model_selection import train_test_split

# this randomly selects and splits the sample into 4 subsets.
# 2 subsets to train and a 2 subsets to test:
x_train, x_test, y_train, y_test = train_test_split(
    digits.data, digits.target, random_state=11 #random_staet for reproducibility
)
print(x_train.shape)
print(x_test.shape)

# estimator model to implement the k-nearest algorithm
# algorithm that is used for machine learning (dont need to know what's "under the hood")
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
# loads training data into the model using the fit method
# Note: the KNeighborsClassifier fit method does not do calculations
# it only loads the model
knn.fit(X=x_train, y=y_train) # only use the train subsets of original data (75% of the original is used to train it)
# often training takes minutes, hours, days, etc.
# more processing power = faster
# more data = longer time for testing

# one way to check if it's doing what we're expecting it to is the predict-method

#Returns an array containing the predicted class of each test image
# creates an array of digits
predicted = knn.predict(X=x_test) # train and test should line up pretty well
expected = y_test

print(predicted[:20])
print(expected[:20])
## they are the same, so it trained correctly

# Want to know where it did not predict correctly:
wrong = [(p,e) for (p,e) in zip(predicted, expected) if p != e]
print(wrong)
# we gave it almost 1400 data values, and it only got around 10 wrong
# if we gave it 10,000+, it should reduce to almost no wrong ones

print(format(knn.score(x_test, y_test), ".2%")) # formats with 2 decimal places
# sows that it got a 97.78% accuracy
# so it got the ones in our lsit comprehension wrong, and that's the ~3%


# NEXT: Confusion matrix
# shows correct and incorrect predicted values
# essentially the hit/misses 

# our model has 10 classes (0-9)
# confusion matrix will show us how many it got correct and incorect for each classes, based on the data we fitted
from sklearn.metrics import confusion_matrix

confusion = confusion_matrix(y_true=expected, y_pred = predicted)
print(confusion)
# numpy array with all hits and misses for each class within the sample
# How to interpret:
# The diagonal line shows the correct ones, causse the classes are 0-9
#First row: 45 hits on 0 (which is the correct one), etc.
# Fourth row had some trouble - it's number 3
# Guessed 5 and 7 incorrectly
# # 8 seems to be the hardest, cause there are the most errors in th 9th row

###################################################
# Now make it look prettier:
# (To actually visualize it with pandas, etc.) 
# Heatmap! 
import pandas as pd # pip install pandas
import seaborn as sns # pipk install seaborn
import matplotlib.pyplot as plt2

confusion_df = pd.DataFrame(confusion, index=range(10), columns=range(10))

figure = plt2.figure(figsize=(7,6))
axes = sns.heatmap(confusion_df, annot=True, cmap= plt2.cm.nipy_spectral_r) 
plt2.show()
print("done")