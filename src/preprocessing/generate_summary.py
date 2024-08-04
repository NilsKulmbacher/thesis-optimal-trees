"""
- This script generates a summary of all datasets
- The summary includes the following columns for each dataset:
    - Name
    - Number of samples
    - Number of features
    - Number of binary features (after binarization)
    - Number of unique values in column y
    - Imbalance ratio (minimum class frequency / maximum class frequency)
- The summary is saved in a csv file
"""

import Datasets as DS
import pandas as pd

# list with all functions from Datasets.py
problems = DS.get_all_problems()

# Loop through all problems and generate summary information
summary = []
for problem in problems:
    pname = problem.__name__.upper()
    print(f'Problem: {pname}')
    df = problem('./datasets/original/')
    df_bin = pd.read_csv(f'./datasets/binary/{pname}_binary.csv')
    features = df.shape[1]-1
    features_bin = df_bin.shape[1]-1
    target = len(df['y'].unique())
    samples = df.shape[0]
    imbalance = df['y'].value_counts().min()/df['y'].value_counts().max()
    # add a new row to the dataframe
    summary.append({'name': pname,'#samples':samples, '#features': features, '#features_bin': features_bin, '#target': target, 'imbalance_ratio': imbalance})

# Save the summary to a csv file
summary = pd.DataFrame(summary)
summary.to_csv('./datasets/datasets_summary.csv', index=False)