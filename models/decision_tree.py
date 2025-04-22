import torch
import numpy as np

class DecisionTreeNode:
    def __init__(self, depth=0, max_depth=3):
        self.left = None
        self.right = None
        self.feature_index = None
        self.threshold = None
        self.value = None
        self.depth = depth
        self.max_depth = max_depth

    def fit(self, X, y):
        # If all y are the same, or max depth reached, stop
        if len(torch.unique(y)) == 1 or self.depth >= self.max_depth:
            self.value = y.mean()
            return

        best_loss = float("inf")
        best_feature, best_thresh = None, None
        best_left, best_right = None, None

        for feature in range(X.shape[1]):
            thresholds = torch.unique(X[:, feature])
            for t in thresholds:
                left_mask = X[:, feature] <= t
                right_mask = X[:, feature] > t
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                y_left, y_right = y[left_mask], y[right_mask]
                loss = (y_left.var() * len(y_left) + y_right.var() * len(y_right)) / len(y)

                if loss < best_loss:
                    best_loss = loss
                    best_feature = feature
                    best_thresh = t
                    best_left = (X[left_mask], y[left_mask])
                    best_right = (X[right_mask], y[right_mask])

        if best_feature is None:
            self.value = y.mean()
            return

        self.feature_index = best_feature
        self.threshold = best_thresh

        self.left = DecisionTreeNode(depth=self.depth + 1, max_depth=self.max_depth)
        self.right = DecisionTreeNode(depth=self.depth + 1, max_depth=self.max_depth)

        self.left.fit(*best_left)
        self.right.fit(*best_right)

    def predict_one(self, x):
        if self.value is not None:
            return self.value.item()
        if x[self.feature_index] <= self.threshold:
            return self.left.predict_one(x)
        else:
            return self.right.predict_one(x)

    def predict(self, X):
        return torch.tensor([self.predict_one(x) for x in X])

# # Sample usage
# torch.manual_seed(42)

# # Fake data: 1 feature, y = x^2 + noise
# X = torch.linspace(-3, 3, 100).reshape(-1, 1)
# y = X.squeeze() ** 2 + torch.randn(X.shape[0]) * 0.3

# tree = DecisionTreeNode(max_depth=4)
# tree.fit(X, y)

# # Predict
# y_pred = tree.predict(X)

# # Plotting (optional, use matplotlib)
# import matplotlib.pyplot as plt
# plt.scatter(X, y, label='True')
# plt.plot(X, y_pred, color='red', label='Tree Prediction')
# plt.legend()
# plt.title("Basic Decision Tree Regression")
# plt.show()
