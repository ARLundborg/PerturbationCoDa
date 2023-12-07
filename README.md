# PerturbationCoDa

This repository contains code for the paper [Perturbation-based Analysis of Compositional Data](https://arxiv.org/abs/2311.18501) by [Anton Rask Lundborg](https://www.arlundborg.com/) and [Niklas Pfister](https://niklaspfister.github.io/).


## Installation
The experiments of the paper are run in Python 3.11. The required packages are specified in the `requirements.txt` file (be aware that the `Cython`, `pandas` and `numpy`-packages need specific versions!). 

To be able to run the code, the `regressiontree` package needs to be installed and compiled. To do so, run the command `pip install -e regressiontree` command. If you have any trouble with this step, feel free to contact one of the authors of the paper via email or open a GitHub issue.

## Code structure
The `main` folder contains the code for the functions used in the experiments while the `experiments` folder contains functions that run the different experiments. The `data` folder holds the two real datasets used in the experiments. The `plots` folder is empty and used to output the results of the Adult data experiment and the semiparametric robustness experiment.

### main
There are five modules in the `main` folder.

- `derivative_estimation` contains the functions used for the nonparametric derivative estimation based on local polynomial smoothing with random forest weights. 
- `perturbation_effects` contains some wrapper-functions for the functions in `semiparametric_estimators` to estimate particular perturbation effects.
- `semiparametric_estimators` contains the primary function calls for the semiparametric estimators used in the experiments.
- `smoothing_spline`contains a python implementation of the `R` smoothing spline functions
- `spline_score` contains functions used for the nonparametric score estimation

### experiments
There are six modules in the `experiments` folder. Some of the modules run full experiments while others are configurable and run a single simulation from an experiment in the paper.

- `sim_adult_experiments` contains the code for the experiment and plot in Section 4.2 of the paper based on the "Adult" dataset. This requires downloading the dataset as described in the data README file in `data/adult`.
- `sim_intro` contains the simulations included in Table 1 of the paper (along with some additional computations that were not included in the paper).
- `sim_microbiome_simple` contains the code for the simple regressions performed in Section 4.3 of the paper. Running the script will create two `.pkl` files. The first, `microbiome-simple.pkl`, contains the marginal effects of L, log-contrast and penalized log-contrast results used to produce Figure 7. The second, `microbiome-pseudocount.pkl`, contains the different log-contrast results when varying pseudocount as shown in the bottom right of Figure 7.
- `sim_microbiome` contains additional code to run simulations for the experiment in Section 4.3. By looping over the `regression`, `measure` and `var_name` variables appropriately, it is possible to reconstruct the results of the paper. Each individual call will produce a `.pkl` file with results. Be aware that the computation time can exceed several hours for a single run. 
- `sim_semiparametric_robustness` contains the code to run the simulations and construct the first two figures in Section S3.1 of the supplementary material of the paper.
- `sim_semiparametric` contains code to run a single instance of the simulations of Section 4.1 of the paper. By looping over the `Y_regression`, `typ`, `n`, `d` and `estimator` variables and repeating this many times, it is possible to recreate Figures 3 and 4. Each call will produce a `.pkl` with results. Be aware that the computation time for a single run can be long when `n` and `d` are large and the estimators are `NPM`.


