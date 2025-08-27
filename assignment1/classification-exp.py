import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification

np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)

# Write the code for Q2 a) and b) below. Show your results.

# Question 2a

X_df = pd.DataFrame(X, columns=["X1", "X2"])
y_df = pd.Series(y, name="y")

# random data split train = 70% and test = 30%
train_size = int(0.7 * len(X))
random_indices = np.random.permutation(len(X))
train_indices = random_indices[:train_size]
test_indices = random_indices[train_size:]

train_X = X_df.iloc[train_indices]
train_y = y_df.iloc[train_indices]
test_X = X_df.iloc[test_indices]
test_y = y_df.iloc[test_indices]

tree = DecisionTree(criterion="information_gain", max_depth=3)
tree.fit(train_X, train_y)
tree.plot()

y_hat = tree.predict(test_X)
print("Accuracy: ", accuracy(y_hat, test_y))
# for class 1

print("Precision for class 1: ", precision(y_hat, test_y, 1))
print("Recall for class 1: ", recall(y_hat, test_y, 1))

# for class 0
print("Precision for class 0: ", precision(y_hat, test_y, 0))
print("Recall for class 0: ", recall(y_hat, test_y, 0))


# Question 2b
