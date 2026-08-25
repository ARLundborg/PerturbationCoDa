import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 16})
plt.rc('text.latex', preamble=r'\usepackage{amsfonts,amssymb,amsthm,amsmath}')
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

true_mu = 1

color1 = "C0"
color2 = "C2"
color3 = "C3"
color4 = "C1"

res_df = pd.read_pickle("experiments/clean_results/semiparametric.pkl")
res_df = res_df[(res_df["n"] == 1000) & (res_df["p"] == 15) & (res_df["Y_regression"] == "partially_linear") &  (res_df["type"] == "binary")]
res_df = res_df.drop(["seed", "n", "p", "Y_regression", "type"], axis=1)
reps = res_df["rep"].nunique()

res_df["estimate"] = res_df["result"].apply(lambda x:x["estimate"])
res_df["standard_error"] = res_df["result"].apply(lambda x:x["standard_error"])
res_df["interval_shift"] = norm.ppf(0.975)*res_df["standard_error"]

res_df["wrong"] = np.abs(res_df["estimate"] - true_mu) > res_df["interval_shift"]
res_df["upper"] = res_df["estimate"] + res_df["interval_shift"]
res_df["lower"] = res_df["estimate"] - res_df["interval_shift"]
res_df = res_df.sort_values(by=["wrong", "estimate"])

res_df_plugin_wrong = res_df[(res_df["estimator"] == "plugin_no_x") & (res_df["wrong"] == True)]
res_df_plugin_right = res_df[(res_df["estimator"] == "plugin_no_x") & (res_df["wrong"] == False)]
res_df_cross_wrong = res_df[(res_df["estimator"] == "onestep_no_x") & (res_df["wrong"] == True)]
res_df_cross_right= res_df[(res_df["estimator"] == "onestep_no_x") & (res_df["wrong"] == False)]
res_df_onestep_wrong = res_df[(res_df["estimator"] == "onestep") & (res_df["wrong"] == True)]
res_df_onestep_right = res_df[(res_df["estimator"] == "onestep") & (res_df["wrong"] == False)]

fig = plt.figure(figsize=(15, 5))
fig.subplots_adjust(wspace=0.4)

ymin = np.min(res_df.loc[res_df["estimator"].isin(["plugin_no_x", "onestep_no_x","onestep"]),"lower"])
ymax = np.max(res_df.loc[res_df["estimator"].isin(["plugin_no_x", "onestep_no_x","onestep"]),"upper"])

ax = fig.add_subplot(131)
ax.set_title("Plug-in estimator \n Coverage = {:.2f}".format(1-res_df[res_df["estimator"] == "plugin_no_x"]["wrong"].mean()))
ax.errorbar(x=np.arange(1, len(res_df_plugin_right)+1), y=res_df_plugin_right["estimate"], yerr=res_df_plugin_right["interval_shift"], color=color3, fmt="none", alpha=0.25)
ax.errorbar(x=np.arange(len(res_df_plugin_right)+1, len(res_df_plugin_right) + len(res_df_plugin_wrong) + 1), y=res_df_plugin_wrong["estimate"], yerr=res_df_plugin_wrong["interval_shift"], color=color3, fmt="none", alpha=1)
ax.hlines(y=true_mu, xmin=1, xmax=reps, linestyles="dashed", color="black")
ax.get_xaxis().set_ticks([])
ax.set_ylim(ymin, ymax)
ax.set_xlabel("Sorted repetition")
ax.set_ylabel(r"Estimate of $\lambda_\psi$")

ax = fig.add_subplot(132)
ax.set_title("One-step estimator \n Coverage = {:.2f}".format(1-res_df[res_df["estimator"] == "onestep_no_x"]["wrong"].mean()))
ax.errorbar(x=np.arange(1, len(res_df_cross_right)+1), y=res_df_cross_right["estimate"], yerr=res_df_cross_right["interval_shift"], color=color1, fmt="none", alpha=0.25)
ax.errorbar(x=np.arange(len(res_df_cross_right)+1, len(res_df_cross_right) + len(res_df_cross_wrong) + 1), y=res_df_cross_wrong["estimate"], yerr=res_df_cross_wrong["interval_shift"], color=color1, fmt="none", alpha=1)
ax.hlines(y=true_mu, xmin=1, xmax=reps, linestyles="dashed", color="black")
ax.get_xaxis().set_ticks([])
ax.set_ylim(ymin, ymax)
ax.set_xlabel("Sorted repetition")
ax.set_ylabel(r"Estimate of $\lambda_\psi$")

ax = fig.add_subplot(133)
ax.set_title("Cross-fit one-step estimator \n Coverage = {:.2f}".format(1-res_df[res_df["estimator"] == "onestep"]["wrong"].mean()))
ax.errorbar(x=np.arange(1, len(res_df_onestep_right)+1), y=res_df_onestep_right["estimate"], yerr=res_df_onestep_right["interval_shift"], color=color2, fmt="none", alpha=0.25)
ax.errorbar(x=np.arange(len(res_df_onestep_right)+1, len(res_df_onestep_right) + len(res_df_onestep_wrong) + 1), y=res_df_onestep_wrong["estimate"], yerr=res_df_onestep_wrong["interval_shift"], color=color2, fmt="none", alpha=1)
ax.hlines(y=true_mu, xmin=1, xmax=reps, linestyles="dashed", color="black")
ax.get_xaxis().set_ticks([])
ax.set_ylim(ymin, ymax)
ax.set_xlabel("Sorted repetition")
ax.set_ylabel(r"Estimate of $\lambda_\psi$")

fig.savefig("plots/confidence_intervals.pdf", bbox_inches="tight")