import pandas as pd
import numpy as np
import main.perturbation_effects as pert
import main.semiparametric_estimators as semi_est
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC
from sklearn.model_selection import GridSearchCV, KFold
from sklearn import clone
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.dummy import DummyClassifier, DummyRegressor
import argparse
import os
import pathlib


parser = argparse.ArgumentParser(
    description="Run microbiome sim using setup file provided",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-i", "--index", type=int,
                    help="row index in the setup data frame to use for the simulation", required=True)
parser.add_argument("-p", "--path", type=str,
                    help="path of the setup file", required=True)
args = vars(parser.parse_args())

setup_path = args.pop("path")
Z_path = os.path.splitext(setup_path)[0] + "-Z.pkl"
Y_path = os.path.splitext(setup_path)[0] + "-Y.pkl"
name = pathlib.Path(setup_path).stem
i = int(args["index"])


sim_setting = pd.read_pickle(setup_path).iloc[i]
Z_df = pd.read_pickle(Z_path)
Y_df = pd.read_pickle(Y_path)

Z = Z_df.to_numpy()
Y = Y_df.to_numpy()

measure = sim_setting["measure"]
regression = sim_setting["regression"]
var_name = sim_setting["var_name"]
if pd.isnull(var_name):
    j = np.nan
else:
    j = np.argmax(Z_df.columns == var_name)
    sim_setting["j"] = j
seed = sim_setting["seed"]
rng = np.random.RandomState(seed)

binary_L_measures = ["CKE", "NP-CKE"]

## Setup regressions
if regression == "rf":
    rf_param_grid = param_grid = [
        {'n_estimators': [250],
         'min_samples_leaf': [1, 0.01],
         'max_depth': [None, 5]}
    ]

    rf = RandomForestRegressor(random_state=rng)

    Y_on_LW = make_pipeline(SelectFromModel(RandomForestRegressor(n_jobs=-1)), GridSearchCV(rf, rf_param_grid, 
    scoring='neg_mean_squared_error', n_jobs=-1))
    if measure in binary_L_measures:
        rf_c = RandomForestClassifier(random_state=rng, criterion="log_loss")
        L_on_W = make_pipeline(SelectFromModel(RandomForestClassifier(n_jobs=-1)), GridSearchCV(rf_c, rf_param_grid, n_jobs=-1, scoring="neg_log_loss"))
    else:
        L_on_W = clone(Y_on_LW)

elif regression == "svr":

    svr_param_grid = [
        {'C': np.geomspace(0.1, 2*10**5, 5),
            'gamma': ["scale"],
            'kernel': ['rbf'],
            "epsilon": [0.0001, 0.01, 0.1]},
    ]

    svc_param_grid = [
        {'C': np.geomspace(0.1, 2*10**5, 5),
            'gamma': ["scale"],
            'kernel': ['rbf']},
    ]

    svr = SVR(max_iter=int(1e6))

    Y_on_LW = make_pipeline(SelectFromModel(RandomForestRegressor(n_jobs=-1)), StandardScaler(), GridSearchCV(svr, svr_param_grid, scoring='neg_mean_squared_error', n_jobs=-1))
    if measure in binary_L_measures:
        svc = SVC(probability=True, random_state=rng, max_iter=int(1e6))
        L_on_W = make_pipeline(SelectFromModel(RandomForestClassifier(n_jobs=-1)), StandardScaler(), GridSearchCV(svc, svc_param_grid, n_jobs=-1, scoring="neg_log_loss"))
    else:
        L_on_W = clone(Y_on_LW)

elif regression == "mlp":

    mlp_param_grid = [
        {'solver': ['adam'],
        'hidden_layer_sizes': [(100,), (20, 20, 20,)],
        'activation': ['relu', "tanh"],
        'alpha': [0.1, 0.01, 0.001]},
    ]
    max_iter = 1000
    mlp = MLPRegressor(max_iter=max_iter, random_state=rng, early_stopping=True)
    Y_on_LW = make_pipeline(SelectFromModel(RandomForestRegressor(n_jobs=-1)), GridSearchCV(mlp, mlp_param_grid, 
    scoring='neg_mean_squared_error', n_jobs=-1))

    if measure in binary_L_measures:
        mlp_c = MLPClassifier(max_iter=max_iter, random_state=rng)
        L_on_W = make_pipeline(SelectFromModel(RandomForestClassifier(n_jobs=-1)), GridSearchCV(mlp_c, mlp_param_grid, n_jobs=-1, scoring="neg_log_loss"))
    else:
        L_on_W = clone(Y_on_LW)


elif regression == "dummy":
    Y_on_LW = make_pipeline(GridSearchCV(DummyRegressor(), param_grid = [{"strategy": ["mean"]}], 
    scoring='neg_mean_squared_error', n_jobs=-1))
    if measure in binary_L_measures:
        L_on_W =make_pipeline(GridSearchCV(DummyClassifier(random_state=rng), param_grid = [{"strategy": ["prior"]}], 
    scoring='neg_log_loss', n_jobs=-1))
    else:
        L_on_W = clone(Y_on_LW)


elif regression == "cv":
    reg_pipe = Pipeline([("scaler", StandardScaler()), 
                          ("estimator", DummyRegressor())])

    reg_parameters = [{
        "estimator": (DummyRegressor(),)
    }, 
    {
        "estimator": (LinearRegression(),)
    },
    {
        "estimator": (RandomForestRegressor(250, n_jobs=-1, random_state=rng),)
    },
    {
        "estimator": (SVR(max_iter=int(1e6)),),
        "estimator__C": np.geomspace(0.1, 2*10**5, 5),
        "estimator__epsilon": [0.1, 0.01, 0.001]
    }
    ]
    Y_on_LW = make_pipeline(GridSearchCV(reg_pipe, reg_parameters, scoring="neg_mean_squared_error"))

    if measure in binary_L_measures:
        clf_pipe = Pipeline([("scaler", StandardScaler()), 
                              ("estimator", DummyClassifier())])

        clf_parameters = [{
            "estimator": (DummyClassifier(),)
        }, 
        {
            "estimator": (LogisticRegression(penalty=None),)
        },
        {
            "estimator": (RandomForestClassifier(250, n_jobs=-1, random_state=rng),)
        },
        {
            "estimator": (SVC(max_iter=int(1e6), probability=True, random_state=rng),),
            "estimator__C": np.geomspace(0.1, 2*10**5, 5),
            "estimator__epsilon": [0.1, 0.01, 0.001]
        }]

        L_on_W = make_pipeline(GridSearchCV(clf_pipe, clf_parameters, scoring="neg_log_loss"))
    else:
        L_on_W = clone(Y_on_LW)        


# Measures

classo_measures = ["classo_2", "classo_8", "classo_32", "classo_2_refit", "classo_8_refit", "classo_32_refit"]


if measure == "perm":
    folds = KFold(5, shuffle=True, random_state=rng).split(Y)
    importances = np.zeros((Z.shape[1], 5))
    for k, (train_idx, test_idx) in enumerate(folds):
        Y_on_LW.fit(Z[train_idx, :], Y[train_idx])
        importances[:, k] = permutation_importance(Y_on_LW, Z[test_idx, :], Y[test_idx], n_repeats=25, n_jobs=-1, random_state=rng)["importances_mean"]
    importances = importances.mean(axis=1)
    importance_results = [{"estimate": importance, "variance": np.nan, "standard_error": np.nan} for importance in importances]
    sim_setting = pd.DataFrame([{"var_name": col_name, "regression": regression, "measure": measure, "j": j, "seed": seed, "result": result} for j, (col_name, result) in enumerate(zip(Z_df.columns, importance_results))])
    sim_setting["Y_reg_score"] = Y_on_LW.fit(Z, Y)[-1].best_score_
    sim_setting["Y_var"] = np.var(Y)
    sim_setting["L_reg_score"] = np.nan
    sim_setting["L_var"] = np.nan
elif measure in classo_measures:
    split_measure = measure.split("_")

    min_denominator = float(split_measure[1])
    refit = (len(split_measure) == 3)
    pseudo_count = np.min(Z[np.where(Z != 0)])/min_denominator
    res = semi_est.sparse_log_contrast(Z, Y, pseudo_count, refit=refit, seed=seed)
    coefs = res["beta"]
    score = res["score"]
    classo_results = [{"estimate": coef, "variance": np.nan, "standard_error": np.nan} for coef in coefs]
    sim_setting = pd.DataFrame([{"var_name": col_name, "regression": regression, "measure": measure, "j": j, "seed": seed, "result": result} for j, (col_name, result) in enumerate(zip(Z_df.columns, classo_results))])
    sim_setting["Y_reg_score"] = score
    sim_setting["Y_var"] = np.var(Y)
    sim_setting["L_reg_score"] = np.nan
    sim_setting["L_var"] = np.nan
else:
    if measure == "DML":
        L = Z[:, j]
        W = np.delete(Z, j, axis=1)
        result = semi_est.partially_linear_model(Y, L, W, Y_on_LW, L_on_W, random_state=rng, folds=5)
    elif measure == "R2":
        L = Z[:, j]
        W = np.delete(Z, j, axis=1)
        result = semi_est.nonparametric_r2(Y, L, W, Y_on_LW, L_on_W, random_state=rng, folds=5)
    elif measure == "CKE":
        L = Z[:, j] == 0
        W = Z.copy()
        W[:, j] = 0
        W = W/(1-Z[:, j])[:, np.newaxis]
        result = pert.cke(Y, Z, j, Y_on_LW, L_on_W, random_state=rng, folds=5)
    elif measure == "NP-CKE":
        L = Z[:, j] == 0
        W = Z.copy()
        W[:, j] = 0
        W = W/(1-Z[:, j])[:, np.newaxis]
        result = pert.cke(Y, Z, j, Y_on_LW, L_on_W, random_state=rng, folds=5, method="nonparametric")
    elif measure == "CFI_mult":
        result = pert.cfi_mult(Y, Z, j, Y_on_LW, L_on_W, random_state=rng, folds=5)
        selected_indices = Z[:, j] > 0

        d = Z.shape[1]
        A_z = np.zeros(d)
        A_z[j] = 1
        norms = np.linalg.norm(A_z - Z[selected_indices, :], axis=1, ord=1)

        L = np.log(Z[selected_indices, j]/(1-Z[selected_indices, j]))
        W = (A_z - Z[selected_indices, :])/norms[:, np.newaxis]
        Y = Y[selected_indices]
    elif measure == "CFI_unit":
        d = Z.shape[1]
        A_z = np.zeros(d)
        A_z[j] = 1
        norms = np.linalg.norm(A_z - Z, axis=1, ord=1)

        L = -norms
        W = (A_z - Z)/norms[:, np.newaxis]
        result = pert.cfi_unit(Y, Z, j, Y_on_LW, L_on_W, random_state=rng, folds=5)
    sim_setting["result"] = result
    sim_setting["L_var"] = np.var(L)
    sim_setting["Y_var"] = np.var(Y)
    sim_setting["Y_reg_score"] = Y_on_LW.fit(W, Y)[-1].best_score_
    sim_setting["L_reg_score"] = L_on_W.fit(W, L)[-1].best_score_



sim_setting.to_pickle(os.path.join(os.path.dirname(setup_path), name + "_" + str(i) + ".pkl"))

