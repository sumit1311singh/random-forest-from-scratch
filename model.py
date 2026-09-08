"""
Random Forest from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impurity
def impurity(labels):
    """Return a non-negative impurity score for a 1D array of integer class labels."""
    # TODO: score how mixed the labels are; 0 for a pure set, larger for more mixed sets.
    n = len(labels)
    if n == 0:
        return 0.0
    
    values, counts = np.unique(labels, return_counts = True)

    if len(values) == 1:
        return 0.0

    prob = counts/n
    prob = prob[prob > 0]

    return -np.sum(prob * np.log(prob))

# Step 2 - split_dataset
import numpy as np

def split_dataset(features, labels, feature_index, threshold):
    # TODO: partition rows into left (feature <= threshold) and right (feature > threshold)
    col = features[:, feature_index]

    mask = col <=  threshold

    left_features = features[mask]
    left_labels = labels[mask]
    right_features = features[~mask]
    right_labels = labels[~mask]

    return (left_features, left_labels, right_features, right_labels)

# Step 3 - split_score
def split_score(parent_labels, left_labels, right_labels):
    # TODO: return a score where higher means the children are purer than the parent.
    n = len(parent_labels)

    w_l = len(left_labels) / n
    w_r = len(right_labels) / n

    return impurity(parent_labels) - (w_l * impurity(left_labels) + w_r * impurity(right_labels))

# Step 4 - best_split
import numpy as np

def best_split(features, labels, feature_indices):
    # TODO: search feature_indices for the (feature, threshold) that best improves purity.

    best_feature_index, best_threshold, best_split_score = None, None, 0.0

    for idx in feature_indices:
        col = np.sort(np.unique(features[:, idx]))

        mid_points = (col[:-1] + col[1:])/2

        for mid_point in mid_points:
            left_features, left_labels, right_features, right_labels = split_dataset(features, labels, idx, mid_point)
            if len(left_labels) == 0 or len(right_labels) == 0:
                continue
            curr_split_score = split_score(labels, left_labels, right_labels)
            if best_split_score < curr_split_score:
                best_split_score = curr_split_score
                best_feature_index = idx
                best_threshold = mid_point
    
    return {
        'feature_index': best_feature_index,
        'threshold': best_threshold,
        'score': best_split_score
        }

# Step 5 - should_stop
def should_stop(labels, depth, max_depth, min_samples_split):
    """Return True if this node should become a leaf instead of splitting further."""
    # TODO: decide whether to stop growing based on purity, depth, and size...
    labels = np.unique(labels)
    if depth >= max_depth or len(labels) == 1 or len(labels) < min_samples_split:
        return True
    return False

# Step 6 - leaf_prediction
def leaf_prediction(labels):
    # TODO: choose a single class label to output for a leaf given the labels that reached it
    values, counts = np.unique(labels, return_counts = True)

    idx = np.argmax(counts)

    return int(values[idx])

# Step 7 - build_tree
def emit_leaf(labels):
    l_pred = leaf_prediction(labels)
    return {
        'leaf': True,
        'prediction': l_pred
    }

def build_tree(features, labels, max_depth=10, min_samples_split=2, feature_subset=None, depth=0):
    # TODO: recursively grow a decision tree, returning a nested dict of leaf/internal nodes.
    if should_stop(labels, depth, max_depth, min_samples_split):
        return emit_leaf(labels)
    
    feature_indices = range(features.shape[1])
    if feature_subset is not None:
        feature_indices = feature_subset

    best_split_dict = best_split(features, labels, feature_indices)
    feature_index = best_split_dict['feature_index']
    threshold = best_split_dict['threshold']

    if feature_index is None:
        return emit_leaf(labels)

    left_features, left_labels, right_features, right_labels = split_dataset(features, labels, feature_index, threshold)

    if len(left_labels) == 0 or len(right_labels) == 0:
        return emit_leaf(labels)

    left_tree = build_tree(left_features, left_labels, max_depth, min_samples_split, feature_subset, depth+1)
    right_tree = build_tree(right_features, right_labels, max_depth, min_samples_split, feature_subset, depth+1)

    return {
        'leaf': False,
        'feature_index': feature_index,
        'threshold': threshold,
        'left': left_tree,
        'right': right_tree
    }

# Step 8 - predict_example_tree
def predict_example_tree(tree, example):
    # TODO: walk the example down the fitted tree until you reach a leaf, then return its prediction.
    if tree['leaf'] is True:
        return int(tree['prediction'])

    feature_idx, threshold = tree['feature_index'], tree['threshold']

    if example[feature_idx] <= threshold:
        return predict_example_tree(tree['left'], example)
    return predict_example_tree(tree['right'], example)

# Step 9 - predict_tree
def predict_tree(tree, features):
    """Predict class labels for every row of `features` using a fitted decision tree.

    tree: dict returned by build_tree
    features: np.ndarray of shape (n, d)
    returns: np.ndarray of shape (n,) with integer class labels
    """
    # TODO: return predicted class for each row of features using the fitted tree.
    tree_predictions = []

    for feature in features:
        tree_predictions.append(int(predict_example_tree(tree, feature)))
    
    return np.array(tree_predictions)

# Step 10 - bootstrap_sample
def bootstrap_sample(features, labels, rng):
    # TODO: draw a bootstrap sample of rows (with replacement) using rng.
    n = features.shape[0]

    sample_idx = rng.integers(0, n, size=n)

    return (features[sample_idx], labels[sample_idx])

# Step 11 - feature_subset
import numpy as np

def feature_subset(num_features, num_to_pick, rng):
    # TODO: return num_to_pick distinct random feature indices from range(num_features) using rng.
    return rng.choice(num_features, num_to_pick, replace=False)

# Step 12 - train_forest
import numpy as np

def train_forest(features, labels, num_trees=10, max_depth=10, min_samples_split=2, num_features_per_split=None, random_state=0):
    # TODO: grow num_trees decision trees on bootstrap samples with random feature subsets.
    rng = np.random.default_rng(random_state)

    num_features = features.shape[1]

    if num_features_per_split is None:
        num_to_pick = int(np.round(np.sqrt(num_features)))
    else:
        num_to_pick = num_features_per_split

    forest = []
    
    for i in range(num_trees):

        bs_features, bs_labels = bootstrap_sample(features, labels, rng)

        feature_subset_col = feature_subset(num_features, num_to_pick, rng)

        tree = build_tree(bs_features, bs_labels, max_depth, min_samples_split, feature_subset_col, 0)

        tree_details = {
            'tree': tree,
            'feature_indices': feature_subset_col
            }

        forest.append(tree_details)

    return forest

# Step 13 - combine_predictions
def combine_predictions(tree_predictions):
    # TODO: aggregate the per-tree predictions of an ensemble into one prediction per example.
    tree_predictions = np.array(tree_predictions)
    
    n = tree_predictions.shape[1]

    predictions = []

    for i in range(n):
        col = tree_predictions[:, i]
    
        counts = np.bincount(col)
    
        prediction = np.argmax(counts)
        predictions.append(prediction)

    return np.array(predictions)

# Step 14 - predict_forest
def predict_forest(forest, features):
    # TODO: predict classes for a dataset using the whole trained forest.
    all_tree_predictions = []

    for tree_details in forest:
        tree = tree_details['tree']
        prediction = predict_tree(tree, features)
        all_tree_predictions.append(prediction)

    return combine_predictions(all_tree_predictions)

# Step 15 - accuracy
def accuracy(predictions, labels):
    # TODO: compute the fraction of entries where predictions equals labels
    return np.mean(predictions == labels)

