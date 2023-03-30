

# %% [markdown]
# # KNN

# %% [markdown]
# ## Imports

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import unittest as ut
import math
from scipy import stats
import sys

# %% [markdown]
# ## Data Loading

train_fname = sys.argv[1]
test_fname = sys.argv[2]

wine_train = pd.read_csv(train_fname, delimiter=" ")
wine_test = pd.read_csv(test_fname, delimiter=" ")
# %% [markdown]
# ## Implement KNN

# %% [markdown]
# The heart of how KNN works by checking the closest k nodes with each feature.
# * Use Euclidean distance to calculate the space between multiple dimensions using the following algorithm taken from https://en.wikipedia.org/wiki/Euclidean_distance    
# * I've added normalization to this formula.
# $$ d(p,q) = \sqrt{\frac{(p_1 - q_1)^2}{(max(p) - min(p))^2} + \frac{(p_2 - q_2)^2}{(max(p) - min(p))^2} + ... + \frac{(p_n - q_n)^2}{(max(p) - min(p))^2}} $$

# %%
class KNN:
    def __init__(self, n_neighbours):
        import numpy as np
        from scipy import stats
        self.n_neighbours = n_neighbours

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        self.max_min = self.__max_min(X)
        return self
    
    def test(self, X_test: pd.DataFrame):
        """Returns the classified nodes in a list."""
        self.classified = [self.__classify(X_test.iloc[row]) for row in range(len(X_test))]
        return self.classified
    
    def __max_min(self, p: pd.DataFrame) -> list[float]:
        """Calculates the difference between the max and min of each predictor variable."""
        self.max_min: list[float] = [(max(p[x]) - min(p[x])) ** 2 for x in p]
        return self.max_min   
    
    def __euclidean(self, ob1: pd.Series, ob2: pd.Series) -> list[float]:
        """Calculates the normalized euclidean distance between all of the features of two observations."""
        return [(ob1[i] - ob2[i])**2 for i in range(len(ob1))]  # type: ignore
    
    def __normalized_euclidean(self, ob1: pd.Series, ob2: pd.Series):
        """Uses the __euclidean function and __max_min function to calculate the normalized euclidean distance between two observations"""
        return np.sqrt(sum(pd.Series(self.__euclidean(ob1, ob2)) / pd.Series(self.max_min)))
    
    def __classify(self, X_test: pd.Series) -> int:
        """Calculate the normalized euclidean distance between each row in X_train and a single X_test row, then sort it with the n closest neighbours and classify."""
        norm_dist = [self.__normalized_euclidean(self.X.iloc[i], X_test) for i in range(len(self.X))]
        sorted_norm_dist = np.sort(norm_dist)[:self.n_neighbours]
        # find where the min k normalized distances are
        locs = [np.where(norm_dist == i) for i in sorted_norm_dist]
        # use these locations to find the corresponding class
        classes = [self.y[i[0][0]] for i in locs]
        # return the majority class
        return stats.mode(classes, keepdims = False)[0]
    
    def accuracy(self, y_test):
        """heck if classified nodes are correct"""
        return np.mean(self.classified == y_test)
    
    def confusion(self, y_test):
        width = len(y_test.unique())
        matrix = np.zeros((width, width))  # row, col - i = true, j = pred
        for i in range(len(self.classified)):
            matrix[y_test[i]-1][self.classified[i]-1] = matrix[y_test[i]-1][self.classified[i]-1] + 1
        self.confusion = matrix # type: ignore
        return self.confusion

# %% [markdown]
# ## Evaluation
# We get an accuracy of 94.38% (2dp) and 95.51% from our KNN model with 1 neighbour and with 3 neighbours respectively.
# From our confusion matrix, we can see:
# * All of the 2nd class was correctly identified. 
# * Our model incorrectly interpreted 2 data points from class 1 as class 2.
# * Incorrectly interpreted 3 data points from class 3 as class 2.
# 
# There appears to be a small amount of bias to class 2.
# 
# Our line plot shows us that the optimal values of K between 1 to 9 inclusive are 3, 7, 8 and 9 with the highest accuracy.

# %%
X_train = wine_train.drop("Class", axis = 1)
y_train = wine_train["Class"]
X_test  = wine_test.drop("Class", axis = 1)
y_test  = wine_test["Class"]    

# %%
myKNN = KNN(1)
pred_y = myKNN.train(X_train, y_train).test(X_test)
print("Accuracy KNN = 1:", myKNN.accuracy(y_test))
print("Confusion Matrix:\n", myKNN.confusion(y_test))
print("Predicted values:",pred_y)

# %%
myKNN = KNN(3)
pred_y = myKNN.train(X_train, y_train).test(X_test)
print("Accuracy KNN = 3:", myKNN.accuracy(y_test))
print("Confusion Matrix:\n", myKNN.confusion(y_test))
print("Predicted values:",pred_y)

