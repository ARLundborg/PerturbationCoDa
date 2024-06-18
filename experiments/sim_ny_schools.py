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


color1 = "C0"
color2 = "C2"
color3 = "C3"

df = pd.read_csv("data/ny-schools/2016 School Explorer.csv")
for column in ['Percent Asian', 'Percent Black', 'Percent Hispanic',"Percent White"]:
  df[column] = df.loc[:, column].str.replace("%", "").astype(float)/100
df = df.loc[df[["Average ELA Proficiency", "Average Math Proficiency"]].isna().sum(axis=1) == 0]
df["School Income Estimate"] = df["School Income Estimate"].str.replace("$", "").str.replace(",", "").astype(float)
df["Community School?"] = df["Community School?"] == "Yes"
df["School Income Quantile"] = pd.qcut(df["School Income Estimate"], 3, ["low", "medium", "high"]).cat.add_categories("missing").fillna("missing")
for column in ["Student Attendance Rate", "Percent of Students Chronically Absent",
               "Rigorous Instruction %", "Collaborative Teachers %",
               "Supportive Environment %", "Effective School Leadership %",
               "Strong Family-Community Ties %", "Trust %", "Percent ELL"]:
  df[column] = df[column].str.replace("%", "").astype(float)/100
df["Grade Low"] = df["Grade Low"].replace({"PK": "-1", "0K": "0"}).astype(int)
df["Grade High"] = df["Grade High"].replace({"PK": "-1", "0K": "0"}).astype(int)

df["School Type"] = None
df.loc[df.loc[:, "Grade Low"] >= 4, "School Type"] = "high only"
df.loc[df.loc[:, "Grade High"] <= 6, "School Type"] = "low only"
df.loc[df["School Type"].isnull(), "School Type"] = "mixed"



Z = df.loc[:, ['Percent Asian','Percent Black', 'Percent Hispanic',"Percent White"]].to_numpy()
Z /= Z.sum(axis=1)[:, np.newaxis]
d = Z.shape[1]

folds = 100
sim_res = list()
Y_columns = ['Average ELA Proficiency', 'Average Math Proficiency']
for column in Y_columns:
  print("Y = {}".format(column))
  L = -np.apply_along_axis(lambda x: np.abs(np.subtract.outer(x, x)).sum(), 1, Z)/(2*d)
  Y = df.loc[:, column].to_numpy()
  for covariates in ["no", "yes"]:
    if covariates == "yes": 
      X = np.c_[pd.get_dummies(df["School Income Quantile"]),
          pd.get_dummies(df["School Type"]), df[["Student Attendance Rate", "Percent of Students Chronically Absent",
               "Rigorous Instruction %", "Collaborative Teachers %",
               "Supportive Environment %", "Effective School Leadership %",
               "Strong Family-Community Ties %", "Trust %", "Economic Need Index", "Community School?", "Percent ELL"]]].astype(float)
    else:
      X = np.zeros((Y.shape[0], 0))

    Y_regression = RandomForestRegressor(250, oob_score=True, max_features=None, random_state=rng, n_jobs=-1)
    L_regression = RandomForestRegressor(250, oob_score=True, max_features=None, random_state=rng, n_jobs=-1)

    if covariates == "yes":
      res = semi_est.partially_linear_model(Y, L, X, Y_regression, L_regression, folds=folds, random_state=rng)
    else:
      res = semi_est.linear_effect(Y, L, np.zeros((L.shape[0], 0)))
    
    if covariates == "yes":
      print("naive_diversity | X, est:{:.3f}, CI: ({:.3f}, {:.3f})".format(res["estimate"], res["estimate"] - 1.96*res["standard_error"], res["estimate"] + 1.96*res["standard_error"]))
    else:
      print("naive_diversity, est:{:.3f}, CI: ({:.3f}, {:.3f})".format(res["estimate"], res["estimate"] - 1.96*res["standard_error"], res["estimate"] + 1.96*res["standard_error"]))

    res = pert.cdi_gini(Y, Z, Y_regression, L_regression, X, folds=folds, random_state=rng)

    if covariates == "yes":
      print("CDI_Gini | X, est:{:.3f}, CI: ({:.3f}, {:.3f})".format(res["estimate"], res["estimate"] - 1.96*res["standard_error"], res["estimate"] + 1.96*res["standard_error"]))
    else:
      print("CDI_Gini, est:{:.3f}, CI: ({:.3f}, {:.3f})".format(res["estimate"], res["estimate"] - 1.96*res["standard_error"], res["estimate"] + 1.96*res["standard_error"]))

