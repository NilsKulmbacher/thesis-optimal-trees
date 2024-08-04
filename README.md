
# BAN Thesis Optimal classification Trees

Repository accomanying bachelor thesis: 
Comparison of four optimal tree formulations with reference models.


## Description

Why? 
Research indicates that optimal decision trees, generated using mathematical optimization formulations, can achieve comparable or better performance than heuristic models while maintaining smaller, more interpretable structures.

What? 
Comprehensive analysis of optimal tree formulations (OCT, BinOCT, FlowOCT, GOSDT) with RandomForest and DecisionTrees. This allows to run all four optimal algorithms under similar circumstances and settings, extracting standardized metrics and plotting trees. Additionally, a class to tune GOSDT trees using different regularization is included. 

### Dependencies

Tested on Python 3.10 with all dependencies listed in `src/requirements.txt` as well as IAI installation and Gurobi solver.

### Installing

- Install requirements - ideally in virtual environment (like [virtualenv](https://virtualenv.pypa.io/en/latest/))
- OCT: [IAI machine key and Julia installation](https://docs.interpretable.ai/stable/IAI-Python/installation/) 
- OCT, BinOCT, FlowOCT: [Gurobi solver license and installation](https://www.gurobi.com/downloads/)
  
### Executing program 

* Set constants in `main.py`
	* *RANDOM_STATE: int*
	* *GLOBAL_MAX_DEPTH: int*
	* *GLOBAL_TIME_LIMIT: int ; in seconds*
	* *OUTPUT_TREES: bool* 
* Specificy datasets to run by setting variable `ds = ...`

* Run all or specify which formulations to run
```
# Running all specificied datasets using all models

if  __name__  ==  '__main__':
	start_tree()
	start_forest()
	start_oct()
	start_bin()
	start_flow()
	start_gosdt()
```

### Project Structure

```
./
├── datasets/ 
│   ├── binary/  *(datasets after binarization)*
│   ├── original/
│   └── datasets_summary.csv
├── src/
│   ├── experiments/ *(notebooks for testing purposes)*
│   ├── modelling/ *(implementations to run the models, metrics, plots)*
│   ├── outputs/ *(default output dir)*
│   │   └── tuning/
│   ├── preprocessing/ *(loading and processing datasets)*
│   ├── results/ *(generating tables and plots from outputs)*
│   │   ├── test_specific_tree.ipynb  *(helper class for validating results)*
│   │   └── analyse_results.ipynb *(main notebook for analysing results)*
│   ├── main.py
│   ├── tuning_gosdt.py *(tuning plots from appendix)*
│   └── requirements.txt
├── thesis.pdf *(finished thesis document)*
└── README.md
```

## Help

BinOCT and FlowOCT:
* Solver runs out of Memory: try modify number of threads in binOCText or StrongTree

OCT:
* Installation not working: specify the correct Julia installation path in PATH sys variable

GOSDT: 
* Taking a long time to compute threshholds: Decrease `max_depth_warm_labels`
* Initialization of problems fails: set config not to use warm labels `reference_LB: False` 

## Authors

Nils Kulmbacher - nils.kulmbacher@student.uva.nl  
  

## Acknowledgments


Aghaei, S., Andrés Gómez, & Vayanos, P. (2021). Strong Optimal Classification Trees. _ArXiv (Cornell University)_. https://doi.org/10.48550/arXiv.2103.15965

Bertsimas, D., & Dunn, J. (2017). Optimal classification trees. _Machine Learning_, _106_(7), 1039–1082. https://doi.org/10.1007/s10994-017-5633-9.

McTavish, H., Zhong, C., Achermann, R., Karimalis, I., Chen, J., Rudin, C., & Seltzer, M. (2021). Fast Sparse Decision Tree Optimization via Reference Ensembles. _ArXiv (Cornell University)_. https://doi.org/10.48550/arxiv.2112.00798

Verwer, S., & Zhang, Y. (2019). Learning Optimal Classification Trees Using a Binary Linear Program Formulation. _Proceedings of the AAAI Conference on Artificial Intelligence_, _33_, 1625–1632. https://doi.org/10.1609/aaai.v33i01.33011624

Locally saved repositories:

 - [StrongTree](https://github.com/D3M-Research-Group/StrongTree)
 - [BinOCT](https://github.com/LucasBoTang/Optimal_Classification_Trees)

Inspiration, code snippets, etc.

* [BAN-thesis-2024](https://github.com/tabearoeber/BAN-thesis-2024) for datasets and pre-processing

