"""
Tuning class for GOSDT model

Output a plot with the accuracy, F1 score and number of nodes for different regularization values and "balance" settings and will be saved in the src/outputs/tuning folder.

How to use:
    - Set the dataset to be used in the variable "problem" (line 31)
    - Optionally, set the configuration for GOSDT and tuning (lines 38-51)
    - Run the script
"""

import seaborn as sns
import preprocessing.Datasets as DS
from preprocessing.Datasets import *
import math
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import graphviz
from sklearn.model_selection import train_test_split
from gosdt.model.threshold_guess import compute_thresholds
from sklearn.metrics import accuracy_score, f1_score
import gosdt
import sys
sys.path.append("..")


# Set the dataset to be used
problem = DS.adult

pname = problem.__name__.upper()
print(f'---{pname}---')
df = problem('./datasets/original/')


# Set the configuration GOSDT and tuning
n_est = 40
d_thresholds = max(1, math.ceil(math.log2(len(np.unique(y))))-1)
config = {
    "depth_budget": 5,
    "reference_LB": False,
    "verbose": False,
    "diagnostics": False,
    "time_limit": 120,
    "allow_small_reg": True
}
min_reg = 1/len(df)
regularization_range = range(0, 20, 1)
balance_settings = [False, True]

# guess thresholds
X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
h = df.columns[:-1]
X = pd.DataFrame(X, columns=h)
print("X:", X.shape)
print("y:", y.shape)
X, thresholds, header, threshold_guess_time = compute_thresholds(
    X, y, n_est, d_thresholds)
y = pd.DataFrame(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Function to find depth for calculating the average depth
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


for balanced in balance_settings:  # Values for balance parameter in GOSDT
    accuracies = []
    f1s = []
    nodes = []

    config["balance"] = balanced
    for i in regularization_range:
        print(f"Regularization: {min_reg * i} with iteration {i}")
        print("Fitting model...")
        config["regularization"] = min_reg * i

        model = gosdt.GOSDT(config)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # convert y_pred: ['False' 'True'] to [False  True] if needed
        if isinstance(y_pred[0], str):
            y_pred = np.array([True if x == 'True' else False for x in y_pred])

        # Save the results for the plot
        num_nodes = model.nodes()
        nodes.append(num_nodes)
        average_depth = (sum(find_prediction_depth(json.loads(
            model.json())))/len(find_prediction_depth(json.loads(model.json()))))-1
        f1 = f1_score(y_test, y_pred, average='macro')*100
        f1s.append(f1)
        accuracies.append(accuracy_score(y_test, y_pred)*100)

    # Plot all the results
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.5)
    sns.set_palette("husl")
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.plot(accuracies, color=color, linestyle='dashed')
    ax1.plot(f1s, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(axis='y')

    ax1.set_ylim([0, 100])
    ax1.set_xticks(regularization_range[::2])
    ax2 = ax1.twinx()  # second axes that shares same x-axis

    color = 'tab:blue'
    ax2.plot(nodes, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.grid(axis='y')

    """ Labelling the plot
    plt.title(f'{pname} - balanced:{config["balance"]} - 1*reg:{min_reg}')
    ax1.set_xlabel('Regularization')
    ax1.set_ylabel('Performance(%)', color=color)
    ax2.set_ylabel('Interpretability (Nodes)', color=color)  
    """

    # Print the numerical results of the tuning
    print(f"Balanced: {balanced}")
    print("________________________________________________")
    print('F1', f1s)
    print("________________________________________________")
    print('Acc', accuracies)
    print("________________________________________________")
    print('Nodes', nodes)
    print("________________________________________________")

    # Show and save the plot
    fig.tight_layout()
    fig.savefig(
        f'./src/outputs/tuning/{pname}_balanced_{config["balance"]}.png')
    plt.show()
