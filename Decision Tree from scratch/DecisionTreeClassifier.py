import numpy as np


class Node:
    def __init__(
        self, feature_index=None, threshold=None, left=None, right=None, value=None
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini_impurity = 1 - np.sum(probabilities**2)
        return gini_impurity

    def split(self, X, y, feature_index, threshold):
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask
        return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

    def find_best_split(self, X, y):
        m, n = X.shape
        if m <= 1:
            return None, None  # Cannot split if there's only one sample

        gini_parent = self.gini(y)
        best_gini = 1
        best_feature_index = None
        best_threshold = None

        for feature_index in range(n):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                X_left, X_right, y_left, y_right = self.split(
                    X, y, feature_index, threshold
                )
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                gini_left = self.gini(y_left)
                gini_right = self.gini(y_right)
                weighted_gini = (len(y_left) / m) * gini_left + (
                    len(y_right) / m
                ) * gini_right

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def build_tree(self, X, y, depth=0):
        if depth == self.max_depth or len(np.unique(y)) == 1:
            unique_classes, counts = np.unique(y, return_counts=True)
            return Node(value=unique_classes[np.argmax(counts)])

        best_feature_index, best_threshold = self.find_best_split(X, y)

        if best_feature_index is None:
            unique_classes, counts = np.unique(y, return_counts=True)
            return Node(value=unique_classes[np.argmax(counts)])

        X_left, X_right, y_left, y_right = self.split(
            X, y, best_feature_index, best_threshold
        )

        left_subtree = self.build_tree(X_left, y_left, depth + 1)
        right_subtree = self.build_tree(X_right, y_right, depth + 1)

        return Node(
            feature_index=best_feature_index,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree,
        )

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict_instance(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self.predict_instance(x, node.left)
        else:
            return self.predict_instance(x, node.right)

    def predict(self, X):
        return [self.predict_instance(x, self.tree) for x in X]
