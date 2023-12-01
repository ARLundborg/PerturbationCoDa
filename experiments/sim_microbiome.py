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
from collections import defaultdict


taxonomic_levels = ["kingdom", "phylum", "class", "order",
                    "family", "genus", "species"]
level_idx = taxonomic_levels.index("species")

# Load data
X = pd.read_csv("data/microbiome/X_data.csv", delimiter=",",
                engine="c", index_col=0, dtype=defaultdict(np.float64, {0: str}))
meta_data = pd.read_csv("data/microbiome/meta_data.csv", delimiter=",", engine="c",
                        index_col=0, na_values=["unknown", "Unspecified"], dtype={0: "str"}, low_memory=False)

BMI = meta_data["weight_kg"].astype(
    float)/(meta_data["height_cm"].astype(float)/100)**2

selected_indices = (BMI > 15) & (BMI < 40)
selected_indices &= (meta_data["height_cm"] > 145) & (
    meta_data["height_cm"] < 220)
selected_indices &= (meta_data["country"] == "USA")
selected_indices &= (meta_data["age_years"] >= 16)

X_num = X.to_numpy()
Z = X_num[selected_indices, :] / \
    X_num[selected_indices, :].sum(axis=1)[:, np.newaxis]
Z = pd.DataFrame(Z, columns=X.columns,
                 index=selected_indices[selected_indices].index)
Y = BMI[selected_indices]

Z_T = Z.T
Z_T["species"] = Z.T.index.str.split(";").map(lambda x: x[6])
Z_T["genus"] = Z.T.index.str.split(";").map(lambda x: x[5])
Z_T["family"] = Z.T.index.str.split(";").map(lambda x: x[4])
Z_T["order"] = Z.T.index.str.split(";").map(lambda x: x[3])
Z_T["class"] = Z.T.index.str.split(";").map(lambda x: x[2])
Z_T["phylum"] = Z.T.index.str.split(";").map(lambda x: x[1])
Z_T["kingdom"] = Z.T.index.str.split(";").map(lambda x: x[0])
Z = Z_T.groupby(taxonomic_levels[0:(level_idx+1)]).sum(numeric_only=True).T
Z.columns = Z.columns.map(";".join)

prevalence_threshold = 0.10
Z = Z.loc[:, (Z > 0).mean(axis=0) > prevalence_threshold]

log_abundance_threshold = -4
Z = Z.loc[:, np.log10(Z.mean(axis=0)) > log_abundance_threshold]

Z = Z.loc[selected_indices, :]/Z.loc[selected_indices, :].sum(axis=1).to_numpy()[:, np.newaxis]


Z_df = Z.copy()
Y_df = Y.copy()

Z = Z_df.to_numpy()
Y = Y_df.to_numpy()

measure = "CKE" # one of ["CKE", "NP-CKE", "CFI_unit", "CFI_mult", "DML", "R2", "perm"]
regression = "rf" # one of ["rf", "mlp", "svr", "cv"]

# var_name holds the name of the target feature (this is ignored by perm that does all variables simultaneously)
var_name = Z_df.index[0]
j = np.argmax(Z_df.columns == var_name)

seed = 11223 # in the experiments this seed was set differently for each single simulation
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
    sim_setting.to_pickle("experiments/{measure}-{reg}.pkl".format(reg=regression, measure=measure))
else:
    sim_setting = pd.Series({"var_name": var_name, "regression": regression, "measure": measure, "j": j, "seed": seed})
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
    sim_setting.to_pickle("experiments/{measure}-{reg}-{j}.pkl".format(reg=regression, measure=measure, j=j))

