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
