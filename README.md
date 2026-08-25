# PerturbationCoDa

This repository contains code for the paper [Perturbation-based Effect Measures for Compositional Data](https://arxiv.org/abs/2311.18501) by [Anton Rask Lundborg](https://www.arlundborg.com/) and [Niklas Pfister](https://niklaspfister.github.io/).


## Installation
The experiments of the paper are run in Python 3.11. The required packages are specified in the `requirements.txt` file (be aware that the `Cython`, `pandas` and `numpy`-packages need specific versions!). 

To be able to run the code, the `regressiontree` package needs to be installed and compiled. To do so, run the command `pip install -e regressiontree` command. If you have any trouble with this step, feel free to contact one of the authors of the paper via email or open a GitHub issue.

The scripts that produce figures typeset their labels with LaTeX (`matplotlib`'s `text.usetex`), so a LaTeX installation providing Computer Modern and the `amsfonts`, `amssymb`, `amsthm` and `amsmath` packages is needed as well. Setting `plt.rcParams["text.usetex"] = False` at the top of a script is enough to run it without LaTeX, at the cost of the typesetting.

## Code structure
The `main` folder contains the code for the functions used in the experiments, the `experiments` folder contains functions that run the different experiments, and the `plot_scripts` folder contains the scripts that turn their output into the figures of the paper. The `data` folder holds one subfolder per dataset used in the experiments; the datasets themselves are not included, and each subfolder has a README describing how to download it. The `plots` folder is empty and used to output the figures.

All scripts are run from the root of the repository with the root on the Python path, for example

```
PYTHONPATH=. python3 experiments/sim_ny_schools.py
```

### main
There are five modules in the `main` folder.

- `derivative_estimation` contains the functions used for the nonparametric derivative estimation based on local polynomial smoothing with random forest weights. 
- `perturbation_effects` contains some wrapper-functions for the functions in `semiparametric_estimators` to estimate particular perturbation effects.
- `semiparametric_estimators` contains the primary function calls for the semiparametric estimators used in the experiments.
- `smoothing_spline`contains a python implementation of the `R` smoothing spline functions
- `spline_score` contains functions used for the nonparametric score estimation

### experiments
There are seven experiment modules in the `experiments` folder. Some of the modules run full experiments while others are configurable and run a single simulation from an experiment in the paper.

- `sim_adult_experiment` contains the code for the experiment and plots in Section 5.1.1 of the paper based on the "Adult" dataset. It prints the estimates of Tables 2 and S2 and writes Figures S9 and S10. This requires downloading the dataset as described in the data README file in `data/adult`.
- `sim_intro` contains the simulations included in Table S1 of the paper (along with some additional computations that were not included in the paper).
- `sim_microbiome_simple` contains the code for the simple regressions performed in Section 5.2 of the paper. Running the script will create two `.pkl` files. The first, `microbiome-simple.pkl`, contains the marginal effects of L, log-contrast and penalized log-contrast results used to produce Figures 2 and S12. The second, `microbiome-pseudocount.pkl`, contains the different log-contrast results when varying pseudocount as shown on the right of Figure 2.
- `sim_microbiome` contains additional code to run simulations for the experiment in Section 5.2. By looping over the `regression`, `measure` and `var_name` variables appropriately, it is possible to reconstruct the results of the paper. Each individual call will produce a `.pkl` file with results. Be aware that the computation time can exceed several hours for a single run. 
- `sim_ny_schools.py` contains the code for the experiment in Section 5.1.2 of the paper based on the PASSNYC data set. It prints the estimates of Table 3. This requires downloading the dataset as described in the data README file in `data/ny-schools`.
- `sim_semiparametric_robustness` contains the code to run the simulations and construct the first two figures in Section S4.4 of the supplementary material of the paper.
- `sim_semiparametric` contains code to run a single instance of the simulations of Section S5.2 of the paper. By looping over the `Y_regression`, `typ`, `n`, `d`, `estimator` and `rep` variables (using a different `seed` for each `rep`), it is possible to recreate Figures 1, S5, S6, S7 and S8. Each call will produce a `.pkl` with results. Be aware that the computation time for a single run can be long when `n` and `d` are large and the estimators are `NPM`.

In addition, `aggregate_results` collects the per-run `.pkl` files written by `sim_microbiome` and `sim_semiparametric` into the combined tables that the plot scripts read, and writes them to `experiments/clean_results`. Run it once the individual runs have finished.

### plot_scripts
There are four modules in the `plot_scripts` folder. They read the results and write the figures to `plots`.

- `plot_perturbations` writes Figure S1. It needs no simulation results and can be run on its own.
- `plot_confidence_intervals` writes Figure 1 from the aggregated `sim_semiparametric` results.
- `plot_semiparametric` writes Figures S5, S6, S7 and S8 from the same aggregated results.
- `plot_microbiome` writes Figures 2, S11, S12 and S13 from the aggregated `sim_microbiome` results together with the two `.pkl` files written by `sim_microbiome_simple`.

So the full workflow for a figure that rests on a simulation study is, for example

```
# one call per setting; see the description of sim_semiparametric above
PYTHONPATH=. python3 experiments/sim_semiparametric.py
PYTHONPATH=. python3 experiments/aggregate_results.py
PYTHONPATH=. python3 plot_scripts/plot_semiparametric.py
```
