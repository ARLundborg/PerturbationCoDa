import numpy as np
import pandas as pd
from collections import defaultdict
import main.semiparametric_estimators as semi_est
import main.perturbation_effects as pert

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


# Run OLS on L
n = Z.shape[0]
def logit(x): return np.log(x/(1-x))


lm_res = [semi_est.linear_effect(Y[x > 0], logit(
    x[x > 0]), np.zeros((np.sum(x > 0), 0))) for x in Z.T.to_numpy()]
lm_res_corrected = [pert.correct_nonzero(
    lm_res[j], x > 0) for j, x in enumerate(Z.T.to_numpy())]
df = pd.DataFrame([{"var_name": name, "regression": "OLS", "measure": "OLS", "j": j, "result": result}
                  for j, (name, result) in enumerate(zip(Z.columns, lm_res_corrected))])

# Run log_contrast
Z_min = np.min(Z.to_numpy()[Z.to_numpy() != 0])
pseudo_count = Z_min/2
log_contrast_res = semi_est.log_contrast(Z+pseudo_count, Y-np.mean(Y))

df = pd.concat((df, pd.DataFrame([{"var_name": name, "regression": "log_contrast", "measure": "log_contrast", "j": j, "result": {
    "estimate": log_contrast_res["beta"][j],
    "variance": log_contrast_res["vcov_mat"][j, j],
    "standard_error": np.sqrt(log_contrast_res["vcov_mat"][j, j]/n)
}} for j, name in enumerate(Z.columns)])))


# Run classo
SEED = 151123
classo_res = semi_est.sparse_log_contrast(Z.to_numpy(), Y.to_numpy() - np.mean(Y), pseudo_count, refit=True, seed=SEED)
df = pd.concat((df, pd.DataFrame([{"var_name": name, "regression": "classo", "measure": "classo", "j": j, "result": {
    "estimate": classo_res["beta"][j],
    "variance": np.nan,
    "standard_error": np.nan
}} for j, name in enumerate(Z.columns)])))

df.to_pickle("experiments/clean_results/microbiome-simple_15-11-23.pkl")


df = pd.DataFrame()
significant_j = [339,  14, 466,  85, 138, 325, 476,  3, 112, 133]
for pseudo_count in 10.0**(-np.linspace(1, 11, 1000)):
    Z_pseudo = (Z+pseudo_count).to_numpy()
    Z_pseudo = Z_pseudo/Z_pseudo.sum(axis=1)[:, np.newaxis]
    log_contrast_res = semi_est.log_contrast(Z_pseudo, Y-np.mean(Y))
    df = pd.concat((df, pd.DataFrame([{"var_name": name, "regression": "log_contrast", "measure": "log_contrast", "L_var": np.var(np.log(Z_pseudo[:, j])),
    "j": j, "pseudo_count": pseudo_count, "result": {
        "estimate": log_contrast_res["beta"][j],
        "variance": log_contrast_res["vcov_mat"][j, j],
        "standard_error": np.sqrt(log_contrast_res["vcov_mat"][j, j]/n)
    }} for j, name in enumerate(Z.columns) if j in significant_j])))

df.to_pickle("experiments/clean_results/microbiome-log-contrast_17-11-23.pkl")


