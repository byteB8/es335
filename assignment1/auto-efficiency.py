import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from metrics import *

np.random.seed(42)

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                   names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                          "acceleration", "model year", "origin", "car name"])

# Clean the above data by removing redundant columns and rows with junk values
# Compare the performance of your model with the decision tree module from scikit learn


data.drop(columns=["car name"], inplace=True)

data.replace('?', np.nan, inplace=True)

data.dropna(inplace=True)

data = data.iloc[:40]  # for testing

X = data.iloc[:, :-1]
y = data.iloc[:, -1]


train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.45, random_state=42)

# convert to pandas
train_X = pd.DataFrame(train_X, columns=["mpg", "cylinders", "displacement", "horsepower", "weight",
                                         "acceleration", "model year", "origin"])
test_X = pd.DataFrame(test_X, columns=["mpg", "cylinders", "displacement", "horsepower", "weight",
                                       "acceleration", "model year", "origin"])
train_y = pd.Series(train_y, name="mpg")


tree = DecisionTree(criterion="information_gain", max_depth=2)
tree.fit(train_X, train_y)

y_hat = tree.predict(test_X)

print("RMSE using my implementation: ", rmse(y_hat, test_y))
print("MAE using my implementation: ", mae(y_hat, test_y))


# With scikit learn

print("###########################")
print("With scikit learn")
print("###########################")

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                   names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                          "acceleration", "model year", "origin", "car name"])


data.drop(columns=["car name"], inplace=True)

data.replace('?', np.nan, inplace=True)

data.dropna(inplace=True)

data = data.iloc[:40]  # for testing

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.45, random_state=42)

# train the model
tree = DecisionTreeRegressor(max_depth=3)
tree.fit(train_X, train_y)

y_hat = tree.predict(test_X)

print("RMSE using scikit learn: ", rmse(y_hat, test_y))
print("MAE using scikit learn: ", mae(y_hat, test_y))
