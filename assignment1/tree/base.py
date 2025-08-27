"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import *

np.random.seed(42)


@dataclass
class DecisionTree:
    # criterion won't be used for regression
    criterion: Literal["information_gain", "gini_index"]
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """

        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree.

        current_depth = 0
        is_discrete = not check_ifreal(y)
        self.root = self._build_tree(
            X, y, current_depth, self.criterion, is_discrete)

    def _build_tree(self, X: pd.DataFrame, y: pd.Series, current_depth: int, criterion: Literal["information_gain", "gini_index"], is_discrete: bool, min_samples_split: int = 2) -> Node:
        if current_depth >= self.max_depth or len(y.unique()) == 1 or len(y) < min_samples_split:
            if is_discrete:
                value, counts = np.unique(y, return_counts=True)
                leaf_value = value[np.argmax(counts)]
                return Node(value=leaf_value)
            else:
                return Node(value=np.mean(y))

        feature_idx, threshold, left_idx, right_idx = self._find_best_split(
            X, y, criterion, is_discrete)
        if feature_idx is None:
            if is_discrete:
                value, counts = np.unique(y, return_counts=True)
                leaf_value = value[np.argmax(counts)]
                return Node(value=leaf_value)
            else:
                return Node(value=np.mean(y))

        left_child = self._build_tree(
            X.iloc[left_idx], y.iloc[left_idx], current_depth + 1, criterion, is_discrete, min_samples_split)
        right_child = self._build_tree(
            X.iloc[right_idx], y.iloc[right_idx], current_depth + 1, criterion, is_discrete, min_samples_split)
        return Node(feature_idx=feature_idx, threshold=threshold, left=left_child, right=right_child)

    def _find_best_split(self, X: pd.DataFrame, y: pd.Series, criterion: Literal["information_gain", "gini_index"], is_discrete: bool) -> tuple[int, float, list, list]:
        best_gain = -np.inf
        best_feature_idx, best_threshold = None, None
        best_left_idx, best_right_idx = None, None

        n_samples, n_features = X.shape

        for feature_idx in range(n_features):
            # discrete feature
            if X.iloc[:, feature_idx].dtype == 'category' or not check_ifreal(X.iloc[:, feature_idx]):
                values = X.iloc[:, feature_idx].unique()
                for mask in range(1, 1 << (len(values) - 1)):
                    binary = bin(mask)[2:]
                    padded_binary = binary.zfill(len(values))
                    subset_list = [int(values[i]) if isinstance(values[i], np.integer) else str(values[i])
                                   for i, bit in enumerate(padded_binary) if bit == '1']
                    left_idx = np.where(
                        X.iloc[:, feature_idx].isin(subset_list))[0]
                    right_idx = np.where(
                        ~X.iloc[:, feature_idx].isin(subset_list))[0]

                    if len(left_idx) == 0 or len(right_idx) == 0:
                        continue

                    if is_discrete:
                        gain = information_gain(
                            y, left_idx, right_idx, criterion)
                    else:
                        gain = mse_reduction(y, left_idx, right_idx)
                    if gain > best_gain:
                        best_gain = gain
                        best_feature_idx = feature_idx
                        best_threshold = subset_list
                        best_left_idx, best_right_idx = left_idx.astype(
                            int).tolist(), right_idx.astype(int).tolist()
            else:
                sorted_idx = np.argsort(X.iloc[:, feature_idx]).tolist()
                # print("debug", sorted_idx)
                # print("hello", sorted_idx[0], sorted_idx[1])
                # print("X", X[sorted_idx[0], feature_idx],
                #       X[sorted_idx[1], feature_idx])

                for j in range(1, n_samples):
                    if X.iloc[sorted_idx[j], feature_idx] != X.iloc[sorted_idx[j-1], feature_idx]:
                        threshold = (
                            X.iloc[sorted_idx[j], feature_idx] + X.iloc[sorted_idx[j-1], feature_idx]) / 2

                        left_idx = np.where(
                            X.iloc[:, feature_idx] <= threshold)[0]
                        right_idx = np.where(
                            X.iloc[:, feature_idx] > threshold)[0]

                        if is_discrete:
                            gain = information_gain(
                                y, left_idx, right_idx, criterion)
                        else:
                            gain = mse_reduction(y, left_idx, right_idx)
                        if gain > best_gain:
                            best_gain = gain
                            best_feature_idx = feature_idx
                            best_threshold = threshold
                            best_left_idx, best_right_idx = left_idx.astype(
                                int).tolist(), right_idx.astype(int).tolist()
        return best_feature_idx, best_threshold, best_left_idx, best_right_idx

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        # Traverse the tree you constructed to return the predicted values for the given test inputs.

        predictions = np.array([self._predict_one(x, self.root)
                               for x in X.itertuples()])
        return pd.Series(predictions, index=X.index)

    def _predict_one(self, x: pd.Series, node: Node) -> int:
        if node.is_leaf():
            return node.value
        if isinstance(node.threshold, (int, float)):
            return self._predict_one(x, node.left) if x[node.feature_idx] <= node.threshold else self._predict_one(x, node.right)
        elif isinstance(node.threshold, list):
            return self._predict_one(x, node.left) if x[node.feature_idx] in node.threshold else self._predict_one(x, node.right)

    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        self._print_tree(self.root, "")

    def _print_tree(self, node, indent):
        if node.is_leaf():
            print(indent + f"Predict: {node.value}")
            return

        # real split
        if isinstance(node.threshold, (int, float)):
            print(indent + f"?({node.feature_idx} <= {node.threshold})")
            print(indent + "  Y:", end=" ")
            self._print_tree(node.left, indent + "    ")
            print(indent + "  N:", end=" ")
            self._print_tree(node.right, indent + "    ")

        #  discrete split
        elif isinstance(node.threshold, list):
            print(indent + f"?({node.feature_idx} in {node.threshold})")
            print(indent + "  Y:", end=" ")
            self._print_tree(node.left, indent + "    ")
            print(indent + "  N:", end=" ")
            self._print_tree(node.right, indent + "    ")
