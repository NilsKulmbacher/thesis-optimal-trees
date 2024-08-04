"""
Author: tabearoeber
Taken from: https://github.com/tabearoeber/BAN-thesis-2024/

- file contains code to read in a couple of datasets
- one-hot encoding is applied to categorical features
"""

import pandas as pd


# List of all datasets
def get_all_problems():
    return [banknote, ILPD, ionosphere, transfusion, liver, tictactoe, wdbc, mammography, diabetes_pima, oilspill,
            phoneme, seeds, wine, glass, ecoli, mushroom, bank_mkt, hearts, musk, magic, student, adult, compas,
            nursery, default, law, attrition, sensorless, skinnonskin]


"""
Unavailable dataset:
def loan(wd):

    #https://www.kaggle.com/datasets/devanshi23/loan-data-2007-2014
    
    df = pd.read_csv(wd + 'loan_data_2007_2014_prepped.csv')
    return df
"""

def banknote(wd):
    """
    1372 x 5
    2 classes
    https://archive.ics.uci.edu/ml/datasets/banknote+authentication
    """
    df = pd.read_csv(wd + 'data_banknote_authentication.csv', header=None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['y']
    return df

def ILPD(wd):
    """
    583 x 10
    2 classes
    https://archive.ics.uci.edu/ml/datasets/ILPD+(Indian+Liver+Patient+Dataset)
    """
    df = pd.read_csv(wd + 'ILPD.csv', header=None)
    df.iloc[:, 1] = (df.iloc[:, 1] == 'Female') * 1
    df.iloc[:, -1] = (df.iloc[:, -1] == 2) * 1
    df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['y']
    df.dropna(inplace=True)
    return df

def ionosphere(wd):
    """
    351 x 34
    2 classes
    https://archive.ics.uci.edu/ml/datasets/ionosphere
    """
    df = pd.read_csv(wd + 'ionosphere.csv', header=None)
    df.iloc[:, -1] = (df.iloc[:, -1] == 'g') * 1
    df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['y']
    df['y'] = df['y'].astype(int)
    return df

def transfusion(wd):
    """
    748 x 5
    2 classes
    https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center
    """
    df = pd.read_csv(wd + 'transfusion.csv', header=0)
    df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['y']
    return df

def liver(wd):
    """
    345 x 7
    2 classes
    https://archive.ics.uci.edu/ml/datasets/liver+disorders
    """
    df = pd.read_csv(wd + 'bupa.csv', header=None)
    df.iloc[:, -1] = (df.iloc[:, -1] == 2) * 1
    df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['y']
    return df

def tictactoe(wd):
    """
    958 x 9
    2 classes
    https://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame
    """
    df = pd.read_csv(wd + 'tictactoe.csv', header=None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['y']
    # df1 = pd.get_dummies(df.iloc[:, :-1], drop_first=True)
    df1 = pd.get_dummies(df.iloc[:, :-1], drop_first=False)
    df1['y'] = (df['y'] == 'positive') * 1
    return df1

def wdbc(wd):  # Two classes
    """
    569 x 31
    2 classes
    https://datahub.io/machine-learning/wdbc
    """
    df = pd.read_csv(wd + 'wdbc.csv', header=None, index_col=0)
    df.columns = ['y'] + ['X_' + str(i) for i in range(len(df.columns) - 1)]
    y = (df['y'] == 'M') * 1
    df.drop('y', axis=1, inplace=True)
    df['y'] = y
    return df

def mammography(wd):
    """
    11183 x 6
    2 classes - Imbalanced
    https://www.openml.org/search?type=data&sort=runs&id=310&status=active
    """
    import pandas as pd
    df = pd.read_csv(wd + 'mammography.csv', header=None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['y']
    return df

def diabetes_pima(wd):
    """
    768 x 8
    2 classes - Imbalanced
    https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
    """
    import pandas as pd
    df = pd.read_csv(wd + 'diabetes_pima.csv', header=None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['y']
    return df

def oilspill(wd):
    """
    937 x 49
    2 classes - Imbalanced
    https://www.kaggle.com/datasets/ashrafkhan94/oil-spill
    """
    import pandas as pd
    df = pd.read_csv(wd + 'oilspill.csv', header=None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['y']
    df = df.drop(df.columns[[0]], axis=1)
    return df

def phoneme(wd):
    """
    5427 x 6
    2 classes - Imbalanced
    https://datahub.io/machine-learning/phoneme
    """
    import pandas as pd
    df = pd.read_csv(wd + 'phoneme.csv', header=None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['y']
    return df

def seeds(wd):
    """
    210 x 7
    3 classes
    https://archive.ics.uci.edu/ml/datasets/seeds
    """
    df = pd.read_csv(wd + 'seeds.csv', header=None, sep='\t', engine='python')
    df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['y']
    return df

def wine(wd):
    """
    178 x 13
    3 classes
    https://archive.ics.uci.edu/ml/datasets/wine
    """
    df = pd.read_csv(wd + 'wine.csv', header=None)
    df.columns = ['y'] + ['X_' + str(i) for i in range(len(df.columns) - 1)]
    y = df['y']
    df.drop('y', axis=1, inplace=True)
    df['y'] = y
    return df

def glass(wd):
    """
    214 x 10
    6 classes - Imbalanced
    https://archive.ics.uci.edu/ml/datasets/glass+identification
    """
    df = pd.read_csv(wd + 'glass.csv', header=None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['y']
    df['y'] -= 1
    return df

def ecoli(wd):
    """
    336 x 8
    8 classes - Imbalanced
    https://archive.ics.uci.edu/ml/datasets/ecoli
    """
    df = pd.read_csv(wd + 'ecoli.csv', header=None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['y']
    return df

def mushroom(wd):
    """
    8214 x 24
    2 classes
    https://archive.ics.uci.edu/ml/datasets/mushroom
    """
    import pandas as pd
    df = pd.read_csv(wd + 'agaricus-lepiota.csv', header=None)
    df.columns = ['y'] + ['X_' + str(i) for i in range(len(df.columns) - 1)]
    # df1 = pd.get_dummies(df.iloc[:, 1:], drop_first=True)
    df1 = pd.get_dummies(df.iloc[:, 1:], drop_first=False)
    df1['y'] = (df['y'] == 'e') * 1
    return df1

def bank_mkt(wd):
    """
    45211 x 17
    2 classes
    https://archive.ics.uci.edu/ml/datasets/bank+marketing
    """

    df = pd.read_csv(wd + 'bank_mkt.csv', header=None)
    y = df.iloc[:, -1]
    df.drop(16, inplace=True, axis=1)
    cols_to_encode = [1, 2, 3, 4, 6, 7, 8, 10, 15]
    # df = pd.get_dummies(data=df, columns=cols_to_encode, drop_first=True)
    df = pd.get_dummies(data=df, columns=cols_to_encode, drop_first=False)
    df.columns = ['X_' + str(i) for i in range(len(df.columns))]
    df['y'] = (y == 'yes') * 1
    return df

def hearts(wd):
    """
    303 x 75
    2 classes
    https://archive.ics.uci.edu/ml/datasets/heart+disease
    """
    df = pd.read_csv(wd + 'hearts.csv', header=None)
    cols_to_encode = [2, 6, 10, 11, 12]
    # df = pd.get_dummies(data=df, columns=cols_to_encode, drop_first=True)
    df = pd.get_dummies(data=df, columns=cols_to_encode, drop_first=False, dtype=int)
    df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['y']
    return df

def musk(wd):
    """
    6589 x 168
    2 classes
    https://archive.ics.uci.edu/ml/datasets/Musk+(Version+2)
    """
    df = pd.read_csv(wd + 'musk.csv', header=None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['y']
    return df

def magic(wd):  # Two classes
    """
    19020 x 11
    2 classes
    https://archive.ics.uci.edu/ml/datasets/magic+gamma+telescope
    """
    import pandas as pd
    df = pd.read_csv(wd + 'magic.csv', header=None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['y']
    df['y'] = (df['y'] == 'g') * 1
    return df


# Fairness Datasets

def student(wd):  # Five classes and two groups, sensitive attribute in the first column
    """
    649 x 33
    The sensitive attribute sex is to be put as the first column.
    https://archive.ics.uci.edu/ml/datasets/student+performance
    """
    df = pd.read_csv(wd + 'student.csv', header=1, sep='\;', engine='python')
    df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['y']
    print('Size data set:', len(df['y']))

    address = df['X_3']
    df = df.drop(columns=['X_3'])
    df.insert(loc=0, column='X_3', value=address)

    return df

def adult(wd):  # Two classes
    """
    48842 x 14
    The sensitive attribute sex is to be put as the first column.
    https://archive.ics.uci.edu/ml/datasets/adult
    """
    df = pd.read_csv(wd + 'adult.csv', header=None)
    y = df.iloc[:, -1]
    df.drop(14, inplace=True, axis=1)
    sex = df[9]
    df = df.drop(columns=[9])
    cols_to_encode = [1, 3, 5, 6, 7, 8, 13]
    # df = pd.get_dummies(data=df, columns=cols_to_encode, drop_first=True)
    df = pd.get_dummies(data=df, columns=cols_to_encode, drop_first=False)
    df.insert(loc=0, column='sex', value=sex)
    df['y'] = (y == ' >50K') * 1
    df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['y']
    print('Size data set:', len(df['y']))

    return df

def compas(wd):
    df = pd.read_csv(wd + 'compas.csv', sep=';')
    y = df['Two_yr_Recidivism']
    df.drop('Two_yr_Recidivism', axis=1, inplace=True)
    df.columns = ['X_' + str(i) for i in range(len(df.columns))]
    df['y'] = y
    df.dropna(inplace=True)
    return df

def nursery(wd):
    """
    12960 x 8
    5 classes
    https://archive.ics.uci.edu/ml/datasets/nursery
    """
    df = pd.read_csv(wd+'nursery.csv', header = None, sep = '\;', engine = 'python')
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    print('Size data set:' , len(df['y']))
    return df

def default(wd):
    """
    30000 x 24
    2 classes
    https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
    """
    df = pd.read_csv(wd + 'default.csv')
    df.dropna(inplace=True)
    cols_to_encode = ['X_2', 'X_3']
    # df = pd.get_dummies(data=df, columns=cols_to_encode, drop_first=True)
    df = pd.get_dummies(data=df, columns=cols_to_encode, drop_first=False)
    return df

def law(wd):  # Five classes
    """
    22387 x 5
    5 classes
    http://www.seaphe.org/databases.php
    """
    df = pd.read_csv(wd + 'law.csv', header=None, sep='\;', engine='python')
    df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['y']
    print('Size data set:', len(df['y']))
    return df

def attrition(wd):
    """
    1469 x 34
    2 classes
    https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
    """
    # Dataset IBM HR analytics employee attrition and performance. 0 = positive class (no attrition), 1=negative class(attrition)
    df = pd.read_csv(wd + 'attrition.csv', header=0, sep='\;', engine='python')
    workLifeBalance = df['WorkLifeBalance']
    df = df.drop(columns=['WorkLifeBalance'])
    df.insert(loc=0, column='X_0', value=workLifeBalance)
    y = df['Attrition']
    df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['y']
    cols_to_encode = ['X_0', 'X_3', 'X_5', 'X_8', 'X_15', 'X_17', 'X_21', 'X_22']
    # df = pd.get_dummies(data=df, columns=cols_to_encode, drop_first=True)
    df = pd.get_dummies(data=df, columns=cols_to_encode, drop_first=False,dtype=int)
    # df['y'] = pd.factorize(df['y'])[0]
    df['y'] = (y == 'Yes') * 1
    # df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['y']
    # for each column print value counts

    return df


## Large Classification

def sensorless(wd):
    """
    58509 x 48
    11 classes
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#Sensorless
    """
    df = pd.read_csv(wd + 'sensorless.csv', header=None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['y']
    df['y'] -= 1
    return df

def skinnonskin(wd):
    """
    245057 x 3
    2 classes
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#skin_nonskin
    """
    df = pd.read_csv(wd + 'skinnonskin.csv', header=None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['y']
    df['y'] -= 1
    return df


