"""
- This file contains functions to extract metrics from the predictions and true values
- There is a function for each type of tree
"""


from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from imblearn.metrics import specificity_score
import numpy as np
import json


def extract_metrics(y_true, y_pred, tree, ensemble=False):
    """
    Extracts metrics from the predictions and true values.

    Parameters
    ----------
    y_true : numpy array
        The true values.
    y_pred : numpy array
        The predicted values.
    ensemble : bool, optional
        If the tree is an ensemble (RF), by default False

    Returns
    -------
    dict
        A dictionary containing the metrics.
    """

    # Helper function to recursively find the depth of each leaf
    def get_leaf_depths(tree):
        leaf_depths = []

        def recurse(node_id, depth, parent):
            if tree.children_left[node_id] == -1:
                leaf_depths.append(depth)
            else:
                if parent != -1:
                    depth += 1
                recurse(tree.children_left[node_id], depth, node_id)
                recurse(tree.children_right[node_id], depth, node_id)

        recurse(0, 0, -1) 
        return leaf_depths

    if ensemble:
        avg_depth = sum(tree.tree_.max_depth for tree in tree.estimators_) / len(tree.estimators_)
        num_leaves = sum(tree.tree_.n_leaves for tree in tree.estimators_) / len(tree.estimators_)
        num_nodes = sum(tree.tree_.node_count for tree in tree.estimators_) / len(tree.estimators_)
        max_depth = max(tree.tree_.max_depth for tree in tree.estimators_)
        is_imbalanced_tree = None
    else:
        avg_depth = 1+np.mean(get_leaf_depths(tree.tree_))
        num_leaves = tree.tree_.n_leaves
        num_nodes = tree.tree_.node_count
        max_depth = tree.tree_.max_depth
        is_imbalanced_tree = (avg_depth!=max_depth)

    features_used = np.count_nonzero(tree.feature_importances_)
    is_multiclass = len(np.unique(y_true)) > 2

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred,zero_division=0) if not is_multiclass else f1_score(y_true, y_pred,average='micro',zero_division=0),
        "auc": roc_auc_score(y_true, y_pred) if not is_multiclass else None,
        'precision': precision_score(y_true, y_pred,zero_division=0) if not is_multiclass else None,
        'recall': recall_score(y_true, y_pred,zero_division=0) if not is_multiclass else None,
        'specificity': specificity_score(y_true, y_pred) if not is_multiclass else None,
        '#leaves': num_leaves,
        '#nodes': num_nodes,
        '#rules': num_nodes-num_leaves,
        'max_depth': max_depth,
        'avg_depth': avg_depth,
        'features_used': features_used,
        'imbalanced': is_imbalanced_tree
    }

def extract_metrics_OCT(y_true, y_pred, learner):
    """
    Extracts metrics from the predictions and true values.

    Parameters
    ----------
    y_true : numpy array
        The true values.
    y_pred : numpy array
        The predicted values.
    learner : OCT
        The OCT model.

    Returns
    -------
    dict
        A dictionary containing the metrics.
    """

    depths = [learner.get_depth(i) for i in range(1,1+learner.get_num_nodes()) if learner.is_leaf(i)]
    avg_depth = np.mean(depths)
    num_leaves = sum([learner.is_leaf(i) for i in range(1,1+learner.get_num_nodes())])
    num_nodes = learner.get_num_nodes()
    max_depth = max(depths)
    is_imbalanced_tree = (avg_depth!=max_depth)
    features_used = len(set([learner.get_split_feature(i) for i in range(1,1+learner.get_num_nodes()) if not learner.is_leaf(i)]))

    is_multiclass = len(np.unique(y_true)) > 2

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred,zero_division=0) if not is_multiclass else f1_score(y_true, y_pred,average='micro',zero_division=0),
        "auc": roc_auc_score(y_true, y_pred) if not is_multiclass else None,
        'precision': precision_score(y_true, y_pred,zero_division=0) if not is_multiclass else None,
        'recall': recall_score(y_true, y_pred,zero_division=0) if not is_multiclass else None,
        'specificity': specificity_score(y_true, y_pred) if not is_multiclass else None,
        '#leaves': num_leaves,
        '#nodes': num_nodes,
        '#rules': num_nodes-num_leaves,
        'max_depth': max_depth,
        'avg_depth': avg_depth,
        'features_used': features_used,
        'imbalanced': is_imbalanced_tree
    }

def extract_metrics_BinOCT(y_true, y_pred, octree):
    """
    Extracts metrics from the predictions and true values.

    Parameters
    ----------
    y_true : numpy array
        The true values.
    y_pred : numpy array
        The predicted values.
    octree : BinOCT
        The BinOCT model.
    
    Returns
    -------
    dict
        A dictionary containing the metrics.
    """

    splits = octree.split
    features_used = [splits[i][0] for i in splits]
    features_used = len(np.unique(features_used))
    num_nodes = len(octree.n_index)
    num_leaves = len(octree.l_index)
    max_depth = octree.max_depth
    avg_depth = octree.max_depth
    is_imbalanced_tree = False

    is_multiclass = len(np.unique(y_true)) > 2

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred,zero_division=0) if not is_multiclass else f1_score(y_true, y_pred,average='micro',zero_division=0),
        "auc": roc_auc_score(y_true, y_pred) if not is_multiclass else None,
        'precision': precision_score(y_true, y_pred,zero_division=0) if not is_multiclass else None,
        'recall': recall_score(y_true, y_pred,zero_division=0) if not is_multiclass else None,
        'specificity': specificity_score(y_true, y_pred) if not is_multiclass else None,
        '#leaves': num_leaves,
        '#nodes': num_nodes,
        '#rules': num_nodes-num_leaves,
        'max_depth': max_depth,
        'avg_depth': avg_depth,
        'features_used': features_used,
        'imbalanced': is_imbalanced_tree
    }

def extract_metrics_FlowOCT(y_true, y_pred,  tree):
    """
    Extracts metrics from the predictions and true values.

    Parameters
    ----------
    y_true : numpy array
        The true values.
    y_pred : numpy array
        The predicted values.
    tree : Tree representation
        The flowOCT_helper Tree representation
    
    Returns
    -------
    dict
        A dictionary containing the metrics.
    """

    num_nodes = tree.num_nodes()
    num_leaves = tree.num_leaves()
    max_depth = tree.max_depth()
    avg_depth = tree.avg_depth()
    is_imbalanced_tree = (max_depth!=avg_depth)
    features_used = tree.features_used()

    is_multiclass = len(np.unique(y_true)) > 2

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred,zero_division=0) if not is_multiclass else f1_score(y_true, y_pred,average='micro',zero_division=0),
        "auc": roc_auc_score(y_true, y_pred) if not is_multiclass else None,
        'precision': precision_score(y_true, y_pred,zero_division=0) if not is_multiclass else None,
        'recall': recall_score(y_true, y_pred,zero_division=0) if not is_multiclass else None,
        'specificity': specificity_score(y_true, y_pred) if not is_multiclass else None,
        '#leaves': num_leaves,
        '#nodes': num_nodes,
        '#rules': num_nodes-num_leaves,
        'max_depth': max_depth,
        'avg_depth': avg_depth,
        'features_used': features_used,
        'imbalanced': is_imbalanced_tree
    }

def extract_metrics_GOSDT(y_true, y_pred, model):
    """
    Extracts metrics from the predictions and true values.

    Parameters
    ----------
    y_true : numpy array
        The true values.
    y_pred : numpy array
        The predicted values.
    model : GOSDT
        The GOSDT model.
    
    Returns
    -------
    dict
        A dictionary containing the metrics.
    """


    json_data = json.loads(model.json())
    nodes = model.tree.source
    num_nodes = model.nodes()
    num_leaves = model.leaves()
    max_depth = model.max_depth()-1

    # Helper function to extract features from the tree using the json representation
    def extract_features_from_name(d):
        features = []

        def recurse_extract(d):
            if isinstance(d, dict):
                if 'name' in d and '<=' in d['name']:
                    feature_name = d['name'].split('<=')[0].strip()
                    features.append(feature_name)
                for key in d:
                    recurse_extract(d[key])

        recurse_extract(d)
        return features

    # Function to recursively find the depth of "prediction" keys, calculate the average depth
    def find_prediction_depth(d, current_depth=0, depths=None):
        if depths is None:
            depths = []
        if isinstance(d, dict):
            for key, value in d.items():
                if key == "prediction":
                    depths.append(current_depth+1)
                else:
                    find_prediction_depth(value, current_depth + 1, depths)
        elif isinstance(d, list):
            for item in d:
                find_prediction_depth(item, current_depth + 1, depths)
        return depths

    depths = find_prediction_depth(json_data)
    features_used = len(set(extract_features_from_name(json_data)))
    avg_depth = (sum(depths) / len(depths))-1
    is_imbalanced_tree = (avg_depth!=max_depth)

    is_multiclass = len(np.unique(y_true)) > 2
    
    # convert y_pred: ['False' 'True'] to [False  True] if needed
    if isinstance(y_pred[0], str):
        y_pred = np.array([True if x == 'True' else False for x in y_pred])

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred,zero_division=0) if not is_multiclass else f1_score(y_true, y_pred,average='micro',zero_division=0),
        "auc": roc_auc_score(y_true, y_pred) if not is_multiclass else None,
        'precision': precision_score(y_true, y_pred,zero_division=0) if not is_multiclass else None,
        'recall': recall_score(y_true, y_pred,zero_division=0) if not is_multiclass else None,
        'specificity': specificity_score(y_true, y_pred) if not is_multiclass else None,
        '#leaves': num_leaves,
        '#nodes': num_nodes,
        '#rules': num_nodes-num_leaves,
        'max_depth': max_depth,
        'avg_depth': avg_depth,
        'features_used': features_used,
        'imbalanced': is_imbalanced_tree
    }