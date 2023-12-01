import numpy as np
import pandas as pd
import main.perturbation_effects as pert
import main.semiparametric_estimators as semi_est
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.colors
import mpltern

seed = 31415
rng = np.random.RandomState(seed)

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 16})
plt.rc('text.latex', preamble=r'\usepackage{amsfonts,amssymb,amsthm,amsmath}')
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

fig = plt.figure(figsize=(12.5, 10))
fig.subplots_adjust(wspace=0.6, hspace=0.2)
color1 = "C0"
color2 = "C2"
color3 = "C3"

def aggregate_df(df, seed, simplex_column_name, group_column_name):
    new_df = df.groupby(group_column_name)[simplex_column_name].apply(lambda x: pd.get_dummies(x).mean(axis=0)).unstack().replace(np.nan, 0)
    
    object_columns = list(df.columns[(df.dtypes == object) & (df.columns != simplex_column_name) & (df.columns != group_column_name)])
    new_df[object_columns]= df.groupby(group_column_name)[object_columns].agg(pd.Series.mode)
    rng = np.random.RandomState(seed)
    for col in object_columns:
        new_df[col] = new_df[col].apply(lambda x: x if len(np.shape(x)) == 0 else rng.choice(x, size=1)[0])
    numeric_columns = list(df.columns[(df.dtypes != object) & (df.columns != group_column_name)])
    new_df[numeric_columns]= df.groupby(group_column_name)[numeric_columns].agg(pd.Series.mean)
    return new_df

# Data loading and cleaning

adult_data_path = 'data/adult/adult.data'
adult_test_path = 'data/adult/adult.test'
cols = ['age','workclass','fnlwgt','education','education-num','marital-status',
        'occupation','relationship','race','sex','capital-gain', 'capital-loss',
        'hours-per-week', 'native-country','compensation']
df_main = pd.read_csv(adult_data_path,
                 names=cols,
                 sep=', ',
                 engine='python')
df_test = pd.read_csv(adult_test_path,
                         skiprows=1,
                         names=cols,
                         sep=', ',
                         engine='python')
df = pd.concat((df_main, df_test)).reset_index(drop=True)
df = df.drop("fnlwgt", axis=1)
df["compensation"] = df["compensation"].replace({'>50K.':'>50K', '<=50K.':'<=50K'})
df["compensation"] = df["compensation"].replace({'>50K':1, '<=50K':0}).astype(float)
df["sex"] = df["sex"].replace({"Male": 0, "Female": 1}).astype(float)
df["race"] = df["race"].replace({"Amer-Indian-Eskimo": "Other", "Asian-Pac-Islander": "Other"})
df[">HS-education"] = (df["education-num"] > 9).astype(int)


# Grouping on edu and race
print("Grouping on race and education:")
## Setting aggregation group probabilities

### P(race)
races = df["race"].unique()
race_prop = (df.groupby("race").size()/len(df)).to_dict()

desired_race_prop_1 = {
    "Black": 0.02,
    "Other": 0.18,
    "White": 0.8
}
inv_race_probs_1 = {race: desired_race_prop_1[race]/race_prop[race] for race in races}
inv_race_probs_sum_1 = sum(inv_race_probs_1.values())
race_prob_dict_1 = {race: inv_race_probs_1[race]/inv_race_probs_sum_1 for race in races}

desired_race_prop_2 = {
    "Black": 0.12,
    "Other": 0.23,
    "White": 0.65,
}
inv_race_probs_2 = {race: desired_race_prop_2[race]/race_prop[race] for race in races}
inv_race_probs_sum_2 = sum(inv_race_probs_2.values())
race_prob_dict_2 = {race: inv_race_probs_2[race]/inv_race_probs_sum_2 for race in races}

### P(edu)
edus = [0, 1]
edu_prop = (df.groupby(">HS-education").size()/len(df)).to_dict()

desired_edu_prop_1 = {
    1: 0.05,
    0: 0.95
}
inv_edu_probs_1 = {edu: desired_edu_prop_1[edu]/edu_prop[edu] for edu in edus}
inv_edu_probs_sum_1 = sum(inv_edu_probs_1.values())
edu_prob_dict_1 = {edu: inv_edu_probs_1[edu]/inv_edu_probs_sum_1 for edu in edus}

desired_edu_prop_2 = {
    1: 0.7,
    0: 0.3
}
inv_edu_probs_2 = {edu: desired_edu_prop_2[edu]/edu_prop[edu] for edu in edus}
inv_edu_probs_sum_2 = sum(inv_edu_probs_2.values())
edu_prob_dict_2 = {edu: inv_edu_probs_2[edu]/inv_edu_probs_sum_2 for edu in edus}


race_edu_prob_dict_1 = {(race, edu): race_prob_dict_1[race]*edu_prob_dict_1[edu] for race in races for edu in edus}

race_edu_prob_dict_2 = {(race, edu): race_prob_dict_2[race]*edu_prob_dict_2[edu] for race in races for edu in edus}

### Aggregating data within group
df["agg_group"] = df.apply(lambda x: rng.choice([0, 1, 2],
            p=np.r_[1-race_edu_prob_dict_1[(x["race"], x[">HS-education"])] - 
                    race_edu_prob_dict_2[(x["race"], x[">HS-education"])],
                    race_edu_prob_dict_1[(x["race"], x[">HS-education"])],
                    race_edu_prob_dict_2[(x["race"], x[">HS-education"])]].astype(float),
            size=1)[0], axis=1)


df = df.sample(frac=1, random_state=rng).reset_index(drop=True)

df["chunk"] = ""
chunk_size = 50 
chunk_index_dict = df.groupby("agg_group").apply(lambda x: np.array_split(x.index, np.ceil(len(x)/chunk_size))).to_dict()
for agg_group, chunk_index_groups in chunk_index_dict.items():
    for i, chunk_index_group in enumerate(chunk_index_groups):
        df.loc[chunk_index_group, "chunk"] = "{}_{}".format(agg_group, i)

final_df = aggregate_df(df, seed, "race", "chunk")


## Plotting

Z = final_df.iloc[:, 0:3].to_numpy()
Y = final_df["compensation"].to_numpy()

d = Z.shape[1]
norms = np.linalg.norm(Z-1/d, axis=1, ord=1)
L = -np.apply_along_axis(lambda x: np.abs(np.subtract.outer(x, x)).sum(), 1, Z)/(2*d)

agg_group = final_df["agg_group"]
ax1 = fig.add_subplot(221, projection="ternary")
ax1.scatter(Z[agg_group == 0, 0], Z[agg_group == 0, 1], Z[agg_group == 0, 2], color=color1,alpha=0.4)
ax1.scatter(Z[agg_group == 1, 0], Z[agg_group == 1, 1], Z[agg_group == 1, 2], color=color2,alpha=0.4)
ax1.scatter(Z[agg_group == 2, 0], Z[agg_group == 2, 1], Z[agg_group == 2, 2], color=color3, alpha=0.4)
ax1.set_title(r"Grouping by race and education", y=1.1)
ax1.set_tlabel(r'Black')
ax1.set_llabel(r'Other')
ax1.set_rlabel(r'White')
ax1.taxis.set_label_position("tick1")
ax1.laxis.set_label_position("tick1")
ax1.raxis.set_label_position("tick1")
ax1.tick_params(axis='both', which='major', labelsize=12)

norms2 = np.linalg.norm(Z-1/d, axis=1)
W = (Z-1/d)/norms2[:, np.newaxis]
A = np.array([[-1/np.sqrt(2), -1/np.sqrt(6)], [1/np.sqrt(2), -1/np.sqrt(6)], [0, 2/np.sqrt(6)]])
B = np.matmul(W, A)
theta = np.sign(B[:, 1])*np.arccos(B[:, 0])

ax2 = fig.add_subplot(223)
ax2.scatter(-L, Y, c=theta, cmap="viridis")
ax2.set_xlabel(r"Gini coefficient")
ax2.set_ylabel(r"Average compensation")

print("Category 0: {}, Category 1: {}, Category 2: {}, Total: {}".format(*final_df["agg_group"].value_counts().to_numpy(), final_df.shape[0]))

## Effect estimation

Y_regression = RandomForestRegressor(250, oob_score=True, max_features=None, random_state=rng, n_jobs=-1)
L_regression = RandomForestRegressor(250, oob_score=True, max_features=None, random_state=rng, n_jobs=-1)
folds = 10


res = pert.cdi_gini(Y, Z, Y_regression, L_regression, folds=folds, random_state=rng)
print("CDI_Gini, est:{:.3f}, CI: ({:.3f}, {:.3f})".format(res["estimate"], res["estimate"] - 1.96*res["standard_error"], res["estimate"] + 1.96*res["standard_error"]))

res = semi_est.linear_effect(Y, L, np.zeros((L.shape[0], 0)))
print("naive_diversity, est:{:.3f}, CI: ({:.3f}, {:.3f})".format(res["estimate"], res["estimate"] - 1.96*res["standard_error"], res["estimate"] + 1.96*res["standard_error"]))

res = semi_est.partially_linear_model(Y, L, Z, Y_regression, L_regression, folds=folds, random_state=rng)
print("naive_diversity | Z, est:{:.3f}, CI: ({:.3f}, {:.3f})".format(res["estimate"], res["estimate"] - 1.96*res["standard_error"], res["estimate"] + 1.96*res["standard_error"]))


X = pd.get_dummies(final_df[["education", "age", "sex"]]).to_numpy()

res = pert.cdi_gini(Y, Z, Y_regression, L_regression, folds=folds, X=X, random_state=rng)
print("CDI_Gini | X, est:{:.3f}, CI: ({:.3f}, {:.3f})".format(res["estimate"], res["estimate"] - 1.96*res["standard_error"], res["estimate"] + 1.96*res["standard_error"]))

res = semi_est.partially_linear_model(Y, L, X, Y_regression, L_regression, folds=folds, random_state=rng)
print("naive_diversity | X, est:{:.3f}, CI: ({:.3f}, {:.3f})".format(res["estimate"], res["estimate"] - 1.96*res["standard_error"], res["estimate"] + 1.96*res["standard_error"]))

res = semi_est.partially_linear_model(Y, L, np.c_[X, Z], Y_regression, L_regression, folds=folds, random_state=rng)
print("naive_diversity | (X, Z), est:{:.3f}, CI: ({:.3f}, {:.3f})".format(res["estimate"], res["estimate"] - 1.96*res["standard_error"], res["estimate"] + 1.96*res["standard_error"]))

# Grouping on comp and race
print("Grouping on race and compensation:")
### P(comp)
comps = [0, 1]
comp_prop = (df.groupby("compensation").size()/len(df)).to_dict()

desired_comp_prop_1 = {
    1: 0.1,
    0: 0.9
}
inv_comp_probs_1 = {comp: desired_comp_prop_1[comp]/comp_prop[comp] for comp in comps}
inv_comp_probs_sum_1 = sum(inv_comp_probs_1.values())
comp_prob_dict_1 = {comp: inv_comp_probs_1[comp]/inv_comp_probs_sum_1 for comp in comps}

desired_comp_prop_2 = {
    1: 0.3,
    0: 0.7
}
inv_comp_probs_2 = {comp: desired_comp_prop_2[comp]/comp_prop[comp] for comp in comps}
inv_comp_probs_sum_2 = sum(inv_comp_probs_2.values())
comp_prob_dict_2 = {comp: inv_comp_probs_2[comp]/inv_comp_probs_sum_2 for comp in comps}


race_comp_prob_dict_1 = {(race, comp): race_prob_dict_1[race]*comp_prob_dict_1[comp] for race in races for comp in comps}

race_comp_prob_dict_2 = {(race, comp): race_prob_dict_2[race]*comp_prob_dict_2[comp] for race in races for comp in comps}

### Aggregating data within group
seed = 838582 + 1
rng = np.random.RandomState(seed)
df["agg_group"] = df.apply(lambda x: rng.choice([0, 1, 2],
            p=np.r_[1-race_comp_prob_dict_1[(x["race"], x["compensation"])] - 
                    race_comp_prob_dict_2[(x["race"], x["compensation"])],
                    race_comp_prob_dict_1[(x["race"], x["compensation"])],
                    race_comp_prob_dict_2[(x["race"], x["compensation"])]].astype(float),
            size=1)[0], axis=1)


df = df.sample(frac=1, random_state=rng).reset_index(drop=True)

df["chunk"] = ""
chunk_size = 50 
chunk_index_dict = df.groupby("agg_group").apply(lambda x: np.array_split(x.index, np.ceil(len(x)/chunk_size))).to_dict()
for agg_group, chunk_index_groups in chunk_index_dict.items():
    for i, chunk_index_group in enumerate(chunk_index_groups):
        df.loc[chunk_index_group, "chunk"] = "{}_{}".format(agg_group, i)

final_df = aggregate_df(df, seed, "race", "chunk")


## Plotting


Z = final_df.iloc[:, 0:3].to_numpy()
Y = final_df["compensation"].to_numpy()

d = Z.shape[1]
norms = np.linalg.norm(Z-1/d, axis=1, ord=1)
L = -np.apply_along_axis(lambda x: np.abs(np.subtract.outer(x, x)).sum(), 1, Z)/(2*d)

agg_group = final_df["agg_group"]
ax3 = fig.add_subplot(222, projection="ternary")
ax3.scatter(Z[agg_group == 0, 0], Z[agg_group == 0, 1], Z[agg_group == 0, 2], color=color1, alpha=0.4)
ax3.scatter(Z[agg_group == 1, 0], Z[agg_group == 1, 1], Z[agg_group == 1, 2], color=color2, alpha=0.4)
ax3.scatter(Z[agg_group == 2, 0], Z[agg_group == 2, 1], Z[agg_group == 2, 2], color=color3, alpha=0.4)
ax3.set_title(r"Grouping by race and compensation", y=1.1)
ax3.set_tlabel(r'Black')
ax3.set_llabel(r'Other')
ax3.set_rlabel(r'White')
ax3.taxis.set_label_position("tick1")
ax3.laxis.set_label_position("tick1")
ax3.raxis.set_label_position("tick1")
ax3.tick_params(axis='both', which='major', labelsize=12)

norms2 = np.linalg.norm(Z-1/d, axis=1)
W = (Z-1/d)/norms2[:, np.newaxis]
A = np.array([[-1/np.sqrt(2), -1/np.sqrt(6)], [1/np.sqrt(2), -1/np.sqrt(6)], [0, 2/np.sqrt(6)]])
B = np.matmul(W, A)
theta = np.sign(B[:, 1])*np.arccos(B[:, 0])

ax4 = fig.add_subplot(224)
colors = ax4.scatter(-L, Y, c=theta, cmap="viridis")
ax4.set_xlabel(r"Gini coefficient")
ax4.set_ylabel(r"Average compensation")
clb = fig.colorbar(colors, ax=[ax2, ax4], fraction=0.1)
clb.ax.set_title(r"$W$")
clb.set_ticks([])
clb = fig.colorbar(ScalarMappable(cmap=matplotlib.colors.ListedColormap(["white"])), ax=[ax1, ax3], fraction=0.1)
clb.outline.set_visible(False)
clb.set_ticks([])

fig.savefig("plots/adult_confounding.pdf", bbox_inches="tight")

print("Category 0: {}, Category 1: {}, Category 2: {}, Total: {}".format(*final_df["agg_group"].value_counts().to_numpy(), final_df.shape[0]))

## Effect estimation

Y_regression = RandomForestRegressor(250, oob_score=True, max_features=None, random_state=rng, n_jobs=-1)
L_regression = RandomForestRegressor(250, oob_score=True, max_features=None, random_state=rng, n_jobs=-1)
folds = 10


res = pert.cdi_gini(Y, Z, Y_regression, L_regression, folds=folds, random_state=rng)
print("CDI_Gini, est:{:.3f}, CI: ({:.3f}, {:.3f})".format(res["estimate"], res["estimate"] - 1.96*res["standard_error"], res["estimate"] + 1.96*res["standard_error"]))

res = semi_est.linear_effect(Y, L, np.zeros((L.shape[0], 0)))
print("naive_diversity, est:{:.3f}, CI: ({:.3f}, {:.3f})".format(res["estimate"], res["estimate"] - 1.96*res["standard_error"], res["estimate"] + 1.96*res["standard_error"]))

res = semi_est.partially_linear_model(Y, L, Z, Y_regression, L_regression, folds=folds, random_state=rng)
print("naive_diversity | Z, est:{:.3f}, CI: ({:.3f}, {:.3f})".format(res["estimate"], res["estimate"] - 1.96*res["standard_error"], res["estimate"] + 1.96*res["standard_error"]))



X = pd.get_dummies(final_df[["education", "age", "sex"]]).to_numpy()

res = pert.cdi_gini(Y, Z, Y_regression, L_regression, folds=folds, X=X, random_state=rng)
print("CDI_Gini| X, est:{:.3f}, CI: ({:.3f}, {:.3f})".format(res["estimate"], res["estimate"] - 1.96*res["standard_error"], res["estimate"] + 1.96*res["standard_error"]))

res = semi_est.partially_linear_model(Y, L, X, Y_regression, L_regression, folds=folds, random_state=rng)
print("naive_diversity | X, est:{:.3f}, CI: ({:.3f}, {:.3f})".format(res["estimate"], res["estimate"] - 1.96*res["standard_error"], res["estimate"] + 1.96*res["standard_error"]))

res = semi_est.partially_linear_model(Y, L, np.c_[X, Z], Y_regression, L_regression, folds=folds, random_state=rng)
print("naive_diversity | (X, Z), est:{:.3f}, CI: ({:.3f}, {:.3f})".format(res["estimate"], res["estimate"] - 1.96*res["standard_error"], res["estimate"] + 1.96*res["standard_error"]))