import pandas as pd
import numpy as np
import main.semiparametric_estimators as semi_est
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, RandomForestClassifier, VotingClassifier
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from scipy.stats import beta
import argparse
import os
import pathlib


def SelectCVRegressor(estimators, random_state=None):
    n_estimators = len(estimators)
    estimators = list(estimators)
    param_grid = {"weights": tuple(np.eye(N=1, M=n_estimators, k=i).flatten() for i in range(n_estimators))}
    return GridSearchCV(VotingRegressor(estimators), scoring="neg_mean_squared_error", param_grid=param_grid, n_jobs=-1, cv=KFold(5, random_state=random_state))

def SelectCVClassifier(estimators, random_state=None):
    n_estimators = len(estimators)
    estimators = list(estimators)
    param_grid = {"weights": tuple(np.eye(N=1, M=n_estimators, k=i).flatten() for i in range(n_estimators))}
    return GridSearchCV(VotingClassifier(estimators, voting="soft"), scoring="neg_log_loss", param_grid=param_grid, n_jobs=-1, cv=StratifiedKFold(5, random_state=random_state))

def normal_score(x, mu, sigma):
    return -(x-mu)/sigma**2


parser = argparse.ArgumentParser(
    description="Run semiparametric sim using setup file provided",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-i", "--index", type=int,
                    help="row index in the setup data frame to use for the simulation", required=True)
parser.add_argument("-p", "--path", type=str,
                    help="path of the setup file", required=True)
args = vars(parser.parse_args())

setup_path = args.pop("path")
name = pathlib.Path(setup_path).stem
i = int(args["index"])


sim_setting = pd.read_pickle(setup_path).iloc[i]


n = sim_setting["n"]
p = sim_setting["p"]
rep = sim_setting["rep"]
Y_regression = sim_setting["Y_regression"]
typ = sim_setting["type"]
seed = sim_setting["seed"]
estimator = sim_setting["estimator"]

rng = np.random.RandomState(seed)

## Simulate data
median = beta.ppf(0.5, 1, p-2)
W = rng.dirichlet(np.ones(p-1), n)
if typ == "continuous":
    xi = rng.normal(0, 1, size=n)
    if Y_regression == "partially_linear":
        true_score = lambda L, W: normal_score(L, (W[:, 0] > median), 1)
        L = (W[:, 0] > median) + xi
        Y = rng.normal(0, 1, n) + L + (W[:, 0] > median)
    elif Y_regression == "nonparametric":
        true_score = lambda L, W: normal_score(L, (W[:, 0] > median), 1+(W[:, 0] > median))
        L = (1 + (W[:,0] > median))*xi + (W[:,0] > median)
        Y = 2*(W[:,0] > median)*L + (W[:, 0] > median) + rng.normal(0,1,n)
elif typ == "binary":
    p_W_0 = 0.8
    p_W_1 = 0.5
    true_score = lambda W:  p_W_1*(W[:,0] > median)+p_W_0*(W[:,0] <= median)
    L = rng.binomial(1, p_W_1, n)*(W[:,0] > median)+rng.binomial(1, p_W_0, n)*(W[:,0] <= median)
    if Y_regression == "nonparametric":
        Y = (1+(1-p_W_0)/(1-p_W_1))*(W[:, 0] > median)*L + W[:, 0]/np.sqrt(beta.var(1, p-2)) + rng.normal(0, 1, n)
    elif Y_regression == "partially_linear":
        Y = L + W[:, 0]/np.sqrt(beta.var(1, p-2))+ rng.normal(0, 1, n)

## Setup regressions
regression_estimators = (("dummy", DummyRegressor()), ("simple_rf", RandomForestRegressor(500, max_features=None, max_depth=2, oob_score=True, random_state=rng, n_jobs=-1)),  ("adv_rf",  RandomForestRegressor(500, oob_score=True, max_features=None, random_state=rng, n_jobs=-1)))
classification_estimators = (("dummy", DummyClassifier()), ("simple_rf", RandomForestClassifier(500, max_features=None, max_depth=2, oob_score=True, random_state=rng, criterion="log_loss", n_jobs=-1)),  ("adv_rf",  RandomForestClassifier(500, oob_score=True, max_features=None, criterion="log_loss", random_state=rng, n_jobs=-1)))

Y_on_LW = SelectCVRegressor(regression_estimators)
if typ == "continuous":
    L_on_W_estimators = regression_estimators
    L_on_W = SelectCVRegressor(L_on_W_estimators)
    conv_var_estimators = regression_estimators
    L_cond_var_W = SelectCVRegressor(conv_var_estimators)
elif typ == "binary":
    L_on_W = SelectCVClassifier(classification_estimators)


## Compute estimates
if estimator == "partially_linear":
    sim_setting["result"] = semi_est.partially_linear_model(Y, L, W, Y_on_LW, L_on_W, random_state=rng)
elif estimator == "partially_linear_no_x":
    sim_setting["result"] = semi_est.partially_linear_model_no_crossfitting(Y, L, W, Y_on_LW, L_on_W)
elif typ == "continuous":
    if estimator == "onestep":
        sim_setting["result"] = semi_est.average_partial_effect(Y, L, W, Y_on_LW, L_on_W, L_cond_var_W, random_state=rng)
    elif estimator == "onestep_true":
        sim_setting["result"] = semi_est.average_partial_effect_true_score(Y, L, W, Y_on_LW, true_score, random_state=rng)
    elif estimator == "onestep_no_x": 
        sim_setting["result"] = semi_est.average_partial_effect_no_crossfitting(Y, L, W, Y_on_LW, L_on_W, L_cond_var_W, random_state=rng)
    elif estimator == "plugin":
        sim_setting["result"] = semi_est.average_partial_effect_plugin(Y, L, W, Y_on_LW, random_state=rng)
    elif estimator == "plugin_no_x":
        sim_setting["result"] = semi_est.average_partial_effect_plugin_no_crossfitting(Y, L, W, Y_on_LW, random_state=rng)
elif typ == "binary":
    if estimator == "onestep":
        sim_setting["result"] = semi_est.average_predictive_effect(Y, L, W, Y_on_LW, L_on_W, random_state=rng)
    elif estimator == "onestep_true":
        sim_setting["result"] = semi_est.average_predictive_effect_true_score(Y, L, W, Y_on_LW, true_score, random_state=rng)
    elif estimator == "onestep_no_x":
        sim_setting["result"] = semi_est.average_predictive_effect_no_crossfitting(Y, L, W, Y_on_LW, L_on_W)
    elif estimator == "plugin":
        sim_setting["result"] = semi_est.average_predictive_effect_plugin(Y, L, W, Y_on_LW, random_state=rng)
    elif estimator == "plugin_no_x":
        sim_setting["result"] = semi_est.average_predictive_effect_plugin_no_crossfitting(Y, L, W, Y_on_LW)

sim_setting.to_pickle(os.path.join(os.path.dirname(setup_path), name + "_" + str(i) + ".pkl"))