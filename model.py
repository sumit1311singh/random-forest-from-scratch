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

# Step 7 - build_tree (not yet solved)
# TODO: implement

# Step 8 - predict_example_tree (not yet solved)
# TODO: implement

# Step 9 - predict_tree (not yet solved)
# TODO: implement

# Step 10 - bootstrap_sample (not yet solved)
# TODO: implement

# Step 11 - feature_subset (not yet solved)
# TODO: implement

# Step 12 - train_forest (not yet solved)
# TODO: implement

# Step 13 - combine_predictions (not yet solved)
# TODO: implement

# Step 14 - predict_forest (not yet solved)
# TODO: implement

# Step 15 - accuracy (not yet solved)
# TODO: implement

