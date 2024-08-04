"""
Main script to run all models on all datasets.

How to use:
    - Set the datasets to be used in the variable "ds" (line 26)
    - Comment out the models that are not to be used (lines 236-243)
    - Optionally, set the configuration for the models (lines 20-23)
    - Run the script
"""

from sklearn.model_selection import train_test_split
from modelling.models import ClassificationTree, OCT, RandomForest, BinOCT, FlowOCT, GOSDT
import preprocessing.Datasets as DS
import traceback
import pandas as pd
import sys
sys.path.append("..")


RANDOM_STATE = 42
GLOBAL_MAX_DEPTH = 5 
GLOBAL_TIME_LIMIT = 600 # Note: only applied to GOSDT, and FlowOCT, BinOCT
OUTPUT_TREES = False

# To run only certain problems, add them to the list below
ds = DS.get_all_problems()
# ds = [DS.problem_name ...]

def start_tree():
    print('STARTING CART...')
    for problem in ds:
        pname = problem.__name__.upper()
        print(f'---{pname}---')
        df = problem('./datasets/original/')

        try:
            # Train test split
            X_train, X_test, y_train, y_test = train_test_split(
                df.drop('y', axis=1), df['y'], test_size=0.2, random_state=RANDOM_STATE)

            # Train and predict with Tree
            tree = ClassificationTree(
                X_train, y_train, pname, max_depth=GLOBAL_MAX_DEPTH, random_state=RANDOM_STATE)
            tree.fit()
            y_pred = tree.predict(X_test)

            # Extract metrics
            metrics = tree.extract_metrics(y_test, y_pred)

            # Save metrics and tree plot
            save_metrics('Tree', metrics, pname)
            if OUTPUT_TREES:
                tree.save_plot_tree()
        except Exception as e:
            print(f'Error: in {pname}')
            # Dump error to file and continue
            save_error('Tree', traceback.format_exc(), pname)
            continue


def start_forest():
    print('STARTING RF...')
    for problem in ds:
        pname = problem.__name__.upper()
        print(f'---{pname}---')
        df = problem('./datasets/original/')

        try:
            # Train test split
            X_train, X_test, y_train, y_test = train_test_split(
                df.drop('y', axis=1), df['y'], test_size=0.2, random_state=RANDOM_STATE)

            # Train and predict with RF
            forest = RandomForest(
                X_train, y_train, pname, max_depth=GLOBAL_MAX_DEPTH, random_state=RANDOM_STATE)
            forest.fit()
            y_pred = forest.predict(X_test)

            # Extract metrics
            metrics = forest.extract_metrics(y_test, y_pred)

            # Save metrics
            save_metrics('RF', metrics, pname)
        except Exception as e:
            print(f'Error: in {pname}')
            # Dump error to file and continue
            save_error('RF', traceback.format_exc(), pname)
            continue


def start_oct():
    print('STARTING OCT...')
    for problem in ds:
        pname = problem.__name__.upper()
        print(f'---{pname}---')
        df = problem('./datasets/original/')

        try:
            # Train test split
            X_train, X_test, y_train, y_test = train_test_split(
                df.drop('y', axis=1), df['y'], test_size=0.2, random_state=RANDOM_STATE, )

            # Train and predict with OCT
            oct = OCT(X_train, y_train, pname,
                      max_depth=GLOBAL_MAX_DEPTH, random_state=RANDOM_STATE)
            oct.fit()
            y_pred = oct.predict(X_test)

            # Extract metrics
            metrics = oct.extract_metrics(y_test, y_pred)

            # Save metrics and tree plot
            save_metrics('OCT', metrics, pname)
            if OUTPUT_TREES:
                oct.save_plot_tree()
        except Exception as e:
            print(f'Error: in {pname}')
            # Dump error to file and continue
            save_error('OCT', traceback.format_exc(), pname)
            continue


def start_bin():
    print('STARTING BinOCT...')
    for problem in ds:
        pname = problem.__name__.upper()
        print(f'---{pname}---')
        # Reading binary datasets manually
        df = pd.read_csv(f'./datasets/binary/{pname}_binary.csv')
        df.columns = ['X_' + str(i)
                      for i in range(len(df.columns) - 1)] + ['y']

        try:
            # Train test split
            X_train, X_test, y_train, y_test = train_test_split(
                df.drop('y', axis=1), df['y'], test_size=0.2, random_state=RANDOM_STATE)

            # Train and predict with BinOCT
            bin = BinOCT(pname, max_depth=GLOBAL_MAX_DEPTH,
                         random_state=RANDOM_STATE)
            bin.fit(X_train, y_train)
            y_pred = bin.predict(X_test)

            # Extract metrics
            metrics = bin.extract_metrics(y_test, y_pred)

            # Save metrics and tree plot
            save_metrics('BinOCT', metrics, pname)
        except Exception as e:
            print(f'Error: in {pname}')
            # Dump error to file and continue
            save_error('BinOCT', traceback.format_exc(), pname)
            continue


def start_flow():
    print('STARTING FlowOCT...')
    for problem in ds:
        pname = problem.__name__.upper()
        print(f'---{pname}---')
        # Reading binary datasets manually
        df = pd.read_csv(f'./datasets/binary/{pname}_binary.csv')
        df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['y']

        try:
            # Train test split
            X_train, X_test, y_train, y_test = train_test_split(
                df.drop('y', axis=1), df['y'], test_size=0.2, random_state=RANDOM_STATE)

            # Train and predict with FlowOCT
            flow = FlowOCT(pname, max_depth=GLOBAL_MAX_DEPTH,
                           random_state=RANDOM_STATE)
            flow.fit(X_train, y_train)
            y_pred = flow.predict(X_test)

            # Extract metrics
            metrics = flow.extract_metrics(y_test, y_pred)

            # Save metrics and tree plot
            save_metrics('FlowOCT', metrics, pname)
            if OUTPUT_TREES:
                flow.save_plot_tree()
        except Exception as e:
            print(f'Error: in {pname}')
            # Dump error to file and continue
            save_error('FlowOCT', traceback.format_exc(), pname)
            continue


def start_gosdt():
    print('STARTING GOSDT...')
    for problem in ds:
        pname = problem.__name__.upper()
        print(f'---{pname}---')
        df = problem('./datasets/original/')

        try:
            # Train and predict with GOSDT
            gosdt = GOSDT(df, pname, max_depth=GLOBAL_MAX_DEPTH,
                          random_state=RANDOM_STATE)
            gosdt.fit()
            y_pred = gosdt.predict()

            # Extract metrics
            metrics = gosdt.extract_metrics(y_pred)

            # Save metrics and tree plot
            save_metrics('GOSDT', metrics, pname)
            if OUTPUT_TREES:
                gosdt.save_plot_tree()
        except Exception as e:
            print(f'Error: in {pname}')
            # Dump error to file and continue
            save_error('GOSDT', traceback.format_exc(), pname)
            continue


# Helper function to append metrics to output file
def save_metrics(model_name, metrics, pname):
    metrics['model'] = model_name
    metrics['problem'] = pname
    metrics['timestamp'] = pd.Timestamp.now()
    pd.DataFrame([metrics]).to_csv("src/outputs/outputs.csv",mode='a', index=False, header=False)


# Helper function to append errors to log file
def save_error(model_name, traceback_output, pname):
    pd.DataFrame([{
        'model': model_name,
        'problem': pname,
        'timestamp': pd.Timestamp.now(),
        'error': traceback_output
    }]).to_csv("src/outputs/errors.csv", mode='a', index=False, header=False)


# Running all specificied datasets using all models
if __name__ == '__main__':
    start_tree()
    start_forest()
    start_oct()
    start_bin()
    start_flow()
    start_gosdt()
