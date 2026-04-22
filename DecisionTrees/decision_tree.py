import numpy as np

def entropy(y):
    classes, count = np.unique(y, return_counts=True)
    probs = count / len(y)
    return -np.sum(probs * np.log2(probs + 1e-9))

def information_gain(y, left_y, right_y):
    l = len(y)
    weighted = (len(left_y)/l)*entropy(left_y) + (len(right_y)/l)*entropy(right_y)
    return entropy(y) - weighted

def best_split(X, y):
    best_ig = -1
    best_feature, best_threshold = None, None
    for feature in range(X.shape[1]):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            left_mask = X[:, feature] <= threshold
            right_mask = ~left_mask
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue
            ig = information_gain(y, y[left_mask], y[right_mask])
            if ig > best_ig:
                best_ig = ig
                best_feature = feature
                best_threshold = threshold
    return best_feature, best_threshold

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def build_tree(X, y, max_depth=None, depth=0):
    if entropy(y) == 0:
        return Node(value=y[0])
    if len(y) == 0:
        return Node(value=0)
    if max_depth is not None and depth >= max_depth:
        return Node(value=np.bincount(y).argmax())
    feature, threshold = best_split(X, y)
    if feature is None:
        return Node(value=np.bincount(y).argmax())
    left_mask = X[:, feature] <= threshold
    right_mask = ~left_mask
    left = build_tree(X[left_mask], y[left_mask], max_depth, depth+1)
    right = build_tree(X[right_mask], y[right_mask], max_depth, depth+1)
    return Node(feature=feature, threshold=threshold, left=left, right=right)

def predict_sample(node, x):
    if node.value is not None:
        return node.value
    if x[node.feature] <= node.threshold:
        return predict_sample(node.left, x)
    else:
        return predict_sample(node.right, x)

def predict(node, X):
    return np.array([predict_sample(node, x) for x in X])

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = build_tree(X, y, self.max_depth)

    def predict(self, X):
        return predict(self.root, X)