import numpy as np


class Node:
    def __init__(self, val=None, feature=None, left=None, right=None):
        self.val = val
        self.feature = None
        self.right = right
        self.left = left
        self.X = None
        self.y = None


class DecisionTreeRegressor:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def mean_squared_error(self, y_true, y_pred):
        # n = len(y_true)  # Number of samples
        # y_mean = np.mean(y_true)  # Mean target value in the node

        # Calculate MSE
        mse = np.mean((y_true - y_pred) ** 2)

        return mse

    def fit(self, X, y, depth=0, node=None):
        if node is None:
            node = Node()

        # Set X and y in the current node
        node.X = X
        node.y = y

        # Check if max depth is reached or if all labels are the same
        if depth == self.max_depth or len(np.unique(y)) == 1:
            node.val = np.mean(y)
            return node

        # Find the best split based on MSE
        best_split_feature, best_split_value, best_mse = self.find_best_split(X, y)

        # If no suitable split is found, assign the mean value to the node
        if best_split_feature is None:
            node.val = np.mean(y)
            return node

        # Perform the split
        left_mask = X[:, best_split_feature] <= best_split_value
        right_mask = ~left_mask

        node.val = best_split_value
        node.feature = best_split_feature
        node.left = self.fit(X[left_mask], y[left_mask], depth + 1, Node())
        node.right = self.fit(X[right_mask], y[right_mask], depth + 1, Node())
        if depth == 0:
            self.tree = node

        return node

    def find_best_split(self, X, y):
        best_split_feature = None
        best_split_value = None
        best_mse = float("inf")

        for feature in range(X.shape[1]):
            unique_values = np.unique(X[:, feature])
            for value in unique_values:
                left_mask = X[:, feature] <= value
                right_mask = ~left_mask

                mse = self.calculate_split_mse(y[left_mask], y[right_mask])

                if mse < best_mse:
                    best_mse = mse
                    best_split_feature = feature
                    best_split_value = value

        return best_split_feature, best_split_value, best_mse

    def calculate_split_mse(self, left_y, right_y):
        left_weight = len(left_y) / (len(left_y) + len(right_y))
        right_weight = len(right_y) / (len(left_y) + len(right_y))

        mse_left = self.mean_squared_error(left_y, np.mean(left_y))
        mse_right = self.mean_squared_error(right_y, np.mean(right_y))

        return left_weight * mse_left + right_weight * mse_right

    def predict(self, X, node=None):
        if node is None:
            node = self.tree

        # If leaf node, return the predicted value for all samples
        if node.left is None and node.right is None:
            return np.full(X.shape[0], node.val)

        # Check the splitting condition and traverse the tree
        mask = X[:, node.feature] <= node.val
        return np.where(mask, self.predict(X, node.left), self.predict(X, node.right))
