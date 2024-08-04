## Datasets

**Documentation by tabearoeber**
**Taken from: https://github.com/tabearoeber/BAN-thesis-2024/**


The original datasets are saved in `../datasets/original/`. A binary version of each file can be found in `../datasets/binary/`. 
There is also a `dataset_info.txt` file which provides a short overview of each dataset, including whether it is a binary or multiclass problem, 
and how many features each version of the dataset has.

The binarization is based on [Lawless et al. (2024)](https://www.jmlr.org/papers/volume24/22-0880/22-0880.pdf). 

## Processing datasets

To process a file, run `main.py`. Here you can specify which dataset(s) to process. 
Note that you need a correctponding function for each dataset in `Datasets.py`, so you'll have to add that when you want to process an entirely new dataset. 
All files provided here are already processed, so there is no need to run this code unless you have a new dataset that you want to use.

## Quickstart to load data

To load a **single dataset**, you can follow this example:

```
import Datasets as DS
import pandas as pd

# load the one-hot encoded version of the dataset
df = DS.adult('../datasets/original/')

# load the binary version of the dataset
df_binary = pd.read_csv('../datasets/binary/ADULT_binary.csv')
```

To go through **several datasets** in a loop, you can follow this example:
```
import Dataset as DS
import pandas as pd

problems = [DS.adult, DS.hearts, DS.banknote]

for problem in problems:
    pname = problem.__name__.upper()
    print(f'---{pname}---')
    df = problem('../datasets/original/')
    df_bin = pd.read_csv(f'../datasets/binary/{pname}_binary.csv')
```


