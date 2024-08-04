"""
Author: tabearoeber
Taken from: https://github.com/tabearoeber/BAN-thesis-2024/

Used binarize datasets and create the training and test splits.
"""

from helpers import *
from Datasets import get_all_problems

numCV: int = 5
testSize = 0.2
randomState = 21
generate_splits_directly = False

# binary classification
problems = get_all_problems()


for problem in problems:
    
    pname = problem.__name__.upper()

    # binarize the dataset and save binarized file in ./datasets/binary/
    binarize_data(problem, save=True)

    # for some methods (e.g. binoct) we need to directly load the training/validation splits as they don't work
    # with sklearn's gridsearch cv
    if generate_splits_directly(problem):
        split_data(problem, binary=True, randomState=randomState, testSize=testSize)
        split_data(problem, binary=False, randomState=randomState, testSize=testSize)

    # get dataset info
    pname = problem.__name__.upper()
    df = problem('./datasets/original/')
    df_bin = pd.read_csv(f'./datasets/binary/{pname}_binary.csv')
    print(f'Problem: {pname}')

    


