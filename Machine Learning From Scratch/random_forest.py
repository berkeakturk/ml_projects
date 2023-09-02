import numpy as np
from decision_tree import DecisionTree
from collections import Counter

class RandomForest():
    def __init__(self, n_trees=20, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            d_tree = DecisionTree(self.min_samples_split, self.max_depth, self.n_features)
            X_sample, y_sample = self._bootstrap_samples(X, y)
            d_tree.fit(X_sample, y_sample)
            self.trees.append(d_tree)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idx = np.random.choice(n_samples, n_samples, replace=True)
        return X[idx], y[idx]

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = [self._most_common_label(pred) for pred in tree_preds]
        return predictions

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    
