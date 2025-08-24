"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np


def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    encoded = []
    columns = X.select_dtypes(include=['object']).columns
    for col in columns:
        categories_unique = list(dict.fromkeys(X[col]))
        categories_index = {cat: idx for idx,
                            cat in enumerate(categories_unique)}
        encoded_col = []
        for cat in X[col]:
            vector = [0] * len(categories_unique)
            vector[categories_index[cat]] = 1
            encoded_col.append(vector)
        encoded_col_df = pd.DataFrame(
            encoded_col, columns=[f"{col}_{cat}" for cat in categories_unique])
        encoded.append(encoded_col_df)
    X = X.drop(columns=columns)
    for i in range(len(encoded)):
        X = pd.concat([X, encoded[i]], axis=1)
    return X


def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    if y.dtype == 'object':
        return False

    perc_unique_count = len(y.unique()) / len(y)
    if perc_unique_count > 0.25:
        return True
    else:
        return False


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    unique_values = Y.unique()
    entropy = 0
    for value in unique_values:
        p = Y[Y == value].shape[0] / Y.shape[0]
        entropy += -p * np.log2(p)
    return entropy


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    unique_values = Y.unique()
    temp = 0
    for value in unique_values:
        p = Y[Y == value].shape[0] / Y.shape[0]
        temp += p*p
    gini_index = 1-temp
    return gini_index


def mse(Y: pd.Series) -> float:
    """
    Function to calculate the mean squared error
    """
    if check_ifreal(Y):
        return np.mean((Y - Y.mean())**2)
    else:
        return 0


def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """
    pass


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).

    pass


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.

    pass


if __name__ == "__main__":
    X = pd.DataFrame({
        'color': ['red', 'blue', 'green', 'red', 'blue', 'green'],
        'size': [1, 2, 3, 4, 5, 6]
    })
    # print(one_hot_encoding(X))
    # print(check_ifreal(X['size']))
    # print(entropy(X['color']))
    # print(gini_index(X['color']))
    print(mse(X['size']))
