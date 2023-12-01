# PerturbationCoDa

This repository contains code for the paper [Perturbation-based Analysis of Compositional Data](https://arxiv.org/abs/2311.18501) by [Anton Rask Lundborg](https://www.arlundborg.com/) and [Niklas Pfister](https://niklaspfister.github.io/).


## Installation
The experiments of the paper are run in Python 3.11. The required packages are specified in the `requirements.txt` file (be aware that the `Cython`-package needs a specific version!). 

To be able to run the code in the package, the `regressiontree` package needs to be installed and compiled. To do so, run the command `pip install -e regressiontree` command. If you have any trouble with this step, feel free to contact one of the authors of the paper via email or open a GitHub issue.

## Code structure
The `main` folder contains the code for the functions used in the experiments while the `experiments` folder contains functions that run the different experiments. The `data` folder holds the two real datasets used in the experiments. The `plots` folder is empty and used to output the results of the Adult data experiment.

### main
There are five modules in the `main` folder.

- `derivative_estimation` contains the functions used for the nonparametric derivative estimation based on local polynomial smoothing with random forest weights. 
- `perturbation_effects` contains some wrapper-functions for the functions in `semiparametric_estimators` to estimate particular perturbation effects.
- `semiparametric_estimators` contains the primary function calls for the semiparametric estimators used in the experiments.
- `smoothing_spline`contains a python implementation of the `R` smoothing spline functions
- `spline_score` contains functions used for the nonparametric score estimation

### Experiments
There are six modules in the `experiments` folder. Some of the modules run full experiments while others are configurable and run a single simulation from an experiment in the paper.

- `sim_adult_experiments` contains the code for the experiment and plot in Section 4.2 of the paper based on the "Adult" dataset. This requires downloading the dataset as described in the data README file in `data/adult`.
- `sim_intro` contains the simulations included in Table 1 of the paper (along with some additional computations that were not included in the paper).
- `sim_microbiome_simple`


