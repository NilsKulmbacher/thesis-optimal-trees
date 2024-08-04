from modelling.flowOCT_helper import make_tree
from modelling.binOCText.binoct import binOptimalDecisionTreeClassifier
from modelling.StrongTree.BendersOCTReplication import *
from gosdt.model.threshold_guess import compute_thresholds
import gosdt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import math
import time
import graphviz

from modelling.metrics import *

# Importing general sklearn functions
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Importing OCT and setting Julia threads
from interpretableai import iai
os.environ['JULIA_NUM_THREADS'] = '4'


class TimeKeeper:
    """
    Class to keep track of time taken by different models
    """

    def __init__(self):
        self.times = {}

    def start(self, model):
        self.times[model] = time.time()

    def end(self, model):
        end_time = time.time()
        elapsed_time = end_time - self.times[model]
        self.times[model] = elapsed_time

    def get_times(self):
        return self.times


class ClassificationTree:
    """
    Implementation for sklearn Classification Tree
    """

    def __init__(self, X_train, y_train, pname, max_depth=5, random_state=42):
        self.X_train = X_train
        self.y_train = y_train
        self.timer = TimeKeeper()
        self.pname = pname
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self):
        self.timer.start('fit_time')
        
        # Grid search for hyperparameter tuning
        params = {'max_depth': range(2, self.max_depth+1),
                  'min_samples_split': [2, 3, 4],
                  'min_samples_leaf': [1, 2]}

        clf = DecisionTreeClassifier(random_state=self.random_state)
        gcv = GridSearchCV(estimator=clf, param_grid=params,
                           scoring='accuracy', cv=3)
        gcv.fit(self.X_train, self.y_train)
        self._model = gcv.best_estimator_
        self._model.fit(self.X_train, self.y_train)
        self.timer.end('fit_time')

    def predict(self, X_test):
        self.timer.start('predict_time')
        r = self._model.predict(X_test)
        self.timer.end('predict_time')
        return r

    def extract_metrics(self, y_true, y_pred):
        metrics = extract_metrics(y_true, y_pred, self._model, ensemble=False)
        metrics.update(self.timer.get_times())
        return metrics

    def save_plot_tree(self):
        fig, ax = plt.subplots(figsize=(20, 20))
        _ = plot_tree(self._model)
        # Save the plot to a file in output folder
        plt.savefig(f'./src/outputs/{self.pname}_basic_tree.png')
        plt.close()


class RandomForest:
    """
    Implementation for sklearn Random Forest
    """

    def __init__(self, X_train, y_train, pname, max_depth=5, random_state=42):
        self.X_train = X_train
        self.y_train = y_train
        self.timer = TimeKeeper()
        self.pname = pname
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self):
        self.timer.start('fit_time')

        # Grid search for hyperparameter tuning
        params = {'max_depth': range(2, self.max_depth+1),
                  'n_estimators': [10, 20, 50, 100],
                  'min_samples_split': [2, 3, 4],
                  'min_samples_leaf': [1, 2]}

        clf = RandomForestClassifier(random_state=self.random_state)
        gcv = GridSearchCV(estimator=clf, param_grid=params,
                           scoring='accuracy', cv=3)
        gcv.fit(self.X_train, self.y_train)
        self._model = gcv.best_estimator_
        self._model.fit(self.X_train, self.y_train)
        self.timer.end('fit_time')

    def predict(self, X_test):
        self.timer.start('predict_time')
        r = self._model.predict(X_test)
        self.timer.end('predict_time')
        return r

    def extract_metrics(self, y_true, y_pred):
        metrics = extract_metrics(y_true, y_pred, self._model, ensemble=True)
        metrics.update(self.timer.get_times())
        return metrics


class OCT:
    """
    Implementation for Optimal Classification Tree (OCT) by Bertsimas and Dunn (2017)
    """

    def __init__(self, X_train, y_train, pname, max_depth=5, random_state=42):
        self.X_train = X_train
        self.y_train = y_train
        self.timer = TimeKeeper()
        self.pname = pname
        self.max_depth = max_depth
        self.random_state = random_state
        self.time_limit = time_limit
        self.grid = None

    def fit(self):
        self.timer.start('fit_time')

        # Build-in grid search for hyperparameter tuning
        self.grid = iai.GridSearch(
            iai.OptimalTreeClassifier(
                random_seed=self.random_state
            ),
            max_depth=range(2, self.max_depth+1),
        )
        self.grid.fit_cv(self.X_train, self.y_train, n_folds=3)
        self.learner = self.grid.get_learner()
        self.timer.end('fit_time')

    def predict(self, X_test):
        self.timer.start('predict_time')
        r = self.grid.predict(X_test)
        self.timer.end('predict_time')
        return r

    def extract_metrics(self, y_true, y_pred):
        metrics = extract_metrics_OCT(y_true, y_pred, self.learner)
        metrics.update(self.timer.get_times())
        return metrics

    def save_plot_tree(self):
        self.learner.write_png(f'./src/outputs/{self.pname}_oct_tree.png')


class BinOCT:
    """
    Implementation for BinOCT Optimal Classification Tree by Verwer and Zhang (2019)
    """

    def __init__(self, pname, max_depth=5, random_state=42, timelimit=600):
        self.pname = pname
        self.max_depth = max_depth
        self.timer = TimeKeeper()
        self.random_state = random_state
        self.timelimit = timelimit

    def fit(self, X_train, y_train):
        self.timer.start('fit_time')

        # Reference model grid search for tuning
        params = {'max_depth': range(2, self.max_depth+1),
                  'min_samples_split': [2, 3, 4],
                  'min_samples_leaf': [1, 2]}

        clf = DecisionTreeClassifier(random_state=self.random_state)
        gcv = GridSearchCV(estimator=clf, param_grid=params,
                           scoring='accuracy', cv=3)
        gcv.fit(X_train, y_train)

        # get best parameter
        optimal_depth = gcv.best_params_['max_depth']
        optimal_min_samples = gcv.best_params_['min_samples_split']

        self._model = binOptimalDecisionTreeClassifier(
            max_depth=optimal_depth, min_samples_split=optimal_min_samples, warmstart=True, timelimit=self.timelimit, output=True)
        self._model.fit(X_train.values, y_train.values)
        self.timer.end('fit_time')

    def predict(self, X_test):
        self.timer.start('predict_time')
        r = self._model.predict(X_test)
        self.timer.end('predict_time')
        return r

    def extract_metrics(self, y_true, y_pred):
        metrics = extract_metrics_BinOCT(y_true, y_pred, self._model)
        metrics.update(self.timer.get_times())
        return metrics


class FlowOCT:
    """
    Implementation for FlowOCT by Aghaei et al. (2021)
    """

    def __init__(self, pname, max_depth=5, random_state=42, timelimit=600):
        self.timer = TimeKeeper()
        self.pname = pname
        self.max_depth = max_depth
        self.random_state = random_state
        self.timelimit = timelimit

    def fit(self, X_train, y_train):
        data_train = pd.concat([X_train, y_train], axis=1)
        self.timer.start('fit_time')

        # Reference model grid search for tuning
        params = {'max_depth': range(2, self.max_depth+1),
                  'min_samples_split': [2, 3, 4],
                  'min_samples_leaf': [1, 2]}

        clf = DecisionTreeClassifier(random_state=self.random_state)
        gcv = GridSearchCV(estimator=clf, param_grid=params,
                           scoring='accuracy', cv=3)
        gcv.fit(X_train, y_train)

        # get best parameter
        self.optimal_depth = gcv.best_params_['max_depth']

        # Regularization parameter [0,1] 
        lambda_val = 0
        
        # Pre-define Benders tree structure
        tree = Tree(self.optimal_depth)
        master = BendersOCT(data_train, 'y', tree,
                            lambda_val, self.timelimit, "classification")

        # Manully create problem and optimize
        master.create_master_problem()
        master.model.update()
        master.model.optimize(mycallback) # custom callback function for BendersCuts (in BendersOCTReplication.py)

        self.timer.end('fit_time')

        b_value = master.model.getAttr("X", master.b)
        beta_value = master.model.getAttr("X", master.beta)
        p_value = master.model.getAttr("X", master.p)
        self.tree = make_tree(master, b_value, beta_value, p_value)

    def predict(self, X_test):
        self.timer.start('predict_time')
        r = self.tree.predict(X_test)
        self.timer.end('predict_time')
        return r

    def extract_metrics(self, y_true, y_pred):
        metrics = extract_metrics_FlowOCT(
            y_true, y_pred, self.tree)
        metrics.update(self.timer.get_times())
        return metrics

    def save_plot_tree(self):
        graph = self.tree.plot_tree()
        graph.render(
            f'./src/outputs/{self.pname}_flowoct_tree', format='png', cleanup=True)
        plt.close()


class GOSDT:
    """
    Implementation for Global Optimal Sparse Decision Trees by McTavish et al. (2021)
    """

    def __init__(self, df, pname, max_depth=5, random_state=42, time_limit=600):
        self.X = df.iloc[:, :-1].values
        columns = df.columns[:-1]
        self.X = pd.DataFrame(self.X, columns=columns)
        self.y = df.iloc[:, -1].values

        self.timelimit = time_limit
        self.timer = TimeKeeper()
        self.pname = pname
        self.max_depth = max_depth
        self.random_state = random_state
        labelpath = "./src/modelling/labels.csv"

        # Define configuration for GOSDT
        self.config = {
            "regularization": 0.05,
            "depth_budget": max_depth+1,  # +1 to account for root node
            "reference_LB": True,
            "verbose": True,
            "diagnostics": True,
            "path_to_labels": labelpath,
            "time_limit": self.timelimit
        }

    # Generate warm labels for GOSDT -> set lower bound for optimization
    def _generate_warm_labels(self, X_train, y_train, n_est=40, max_depth=1):
        print("Generating warm labels...")
        clf = GradientBoostingClassifier(
            n_estimators=n_est, max_depth=max_depth, random_state=42)
        clf.fit(X_train, y_train.values.flatten())
        warm_labels = clf.predict(X_train)
        pd.DataFrame(warm_labels, columns=["class_labels"]).to_csv(
            "./src/modelling/labels.csv", header="class_labels", index=None)

    def fit(self):
        self.timer.start('fit_time')

        # Parameters for threshold guessing
        n_est = 40
        max_depth_warm_labels = max(1, math.ceil(
            math.log2(len(np.unique(self.y))))-1)

        # Compute thresholds for GOSDT
        self.X, thresholds, header, threshold_guess_time = compute_thresholds(
            self.X, self.y, n_est, max_depth_warm_labels)
        self.y = pd.DataFrame(self.y)

        X_train, self.X_test, y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42)

        if self.config["reference_LB"]:
            self._generate_warm_labels(X_train, y_train, n_est=n_est)

        self._model = gosdt.GOSDT(self.config)
        self._model.fit(X_train, y_train)
        self.timer.end('fit_time')
        if self.config["reference_LB"]:
            os.remove("./src/modelling/labels.csv")
        self.json_tree = json.loads(self._model.json())

    def predict(self):
        self.timer.start('predict_time')
        r = self._model.predict(self.X_test)
        self.timer.end('predict_time')
        return r

    def extract_metrics(self, y_pred):
        metrics = extract_metrics_GOSDT(self.y_test, y_pred, self._model)
        metrics.update(self.timer.get_times())
        return metrics

    def _make_tree_plot(self, json_data, graph=None, parent=None, edge_label=''):
        if graph is None:
            graph = graphviz.Digraph()

        node_label = f'{json_data["name"]}'
        if 'prediction' in json_data:
            node_label += f'\nPrediction: {json_data["prediction"]}'

        node_id = id(json_data)
        graph.node(name=str(node_id), label=node_label)

        if parent is not None:
            graph.edge(str(parent), str(node_id), label=edge_label)

        if 'true' in json_data:
            self._make_tree_plot(
                json_data['true'], graph, parent=node_id, edge_label='True')
        if 'false' in json_data:
            self._make_tree_plot(
                json_data['false'], graph, parent=node_id, edge_label='False')

        return graph

    def save_plot_tree(self):
        graph = self._make_tree_plot(self.json_tree)
        graph.render(
            f'./src/outputs/{self.pname}_gosdt_tree', format='png', cleanup=True)
        plt.close()
