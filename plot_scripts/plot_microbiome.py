import os

import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from scipy.stats import norm, spearmanr


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({name},{a:.2f},{b:.2f})".format(
            name=cmap.name, a=minval, b=maxval
        ),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


colormap = truncate_colormap(mpl.colormaps["viridis"], 0.2, 0.9)


plt.rcParams["text.usetex"] = True
plt.rcParams.update({"font.size": 16})
plt.rc("text.latex", preamble=r"\usepackage{amsfonts,amssymb,amsthm,amsmath}")
plt.rc("font", **{"family": "serif", "serif": ["Computer Modern"]})

color1 = "C0"
color2 = "C2"
color3 = "C3"
color4 = "C1"
color5 = "C5"

color_list = [color1, color2, color3, color4]

res_df = pd.read_pickle(
    "experiments/clean_results/microbiome-comparison.pkl"
)

binary_measures = ["CKE", "NP-CKE"]

res_df["variance"] = res_df["result"].apply(lambda x: x["variance"])
res_df["estimate"] = res_df["result"].apply(lambda x: x["estimate"])
res_df["scaled_estimate"] = res_df["estimate"] / np.sqrt(res_df["variance"])
res_df["standard_error"] = res_df["result"].apply(lambda x: x["standard_error"])
res_df["lower_ci"] = res_df["result"].apply(
    lambda x: x["estimate"] - 1.96 * x["standard_error"]
)
res_df["upper_ci"] = res_df["result"].apply(
    lambda x: x["estimate"] + 1.96 * x["standard_error"]
)
res_df["p_value"] = 2 * norm.cdf(
    -np.abs(res_df["estimate"] / res_df["standard_error"])
)
res_df["significant"] = res_df["p_value"] < 0.05
res_df = res_df.drop(["result", "seed"], axis=1)
res_df["L_reg_score_L_var"] = res_df["L_reg_score"] / res_df["L_var"]
res_df["Y_reg_score_Y_var"] = res_df["Y_reg_score"] / res_df["Y_var"]


##### Correlation plot #####

correlation_df = pd.DataFrame()
correlation_measures = ["CFI_mult", "CFI_unit", "CKE", "NP-CKE", "perm", "R2"]
for measure in correlation_measures:
    measure_df = res_df[res_df["measure"] == measure].sort_values("j")
    correlation_df = pd.concat(
        (
            correlation_df,
            pd.DataFrame(
                {
                    "correlation": spearmanr(
                        measure_df[measure_df["regression"] == "rf"][
                            "estimate"
                        ],
                        measure_df[measure_df["regression"] == "mlp"][
                            "estimate"
                        ],
                    )[0],
                    "pair": "rf-mlp",
                    "measure": measure,
                },
                index=[0],
            ),
        )
    ).reset_index(drop=True)
    correlation_df = pd.concat(
        (
            correlation_df,
            pd.DataFrame(
                {
                    "correlation": spearmanr(
                        measure_df[measure_df["regression"] == "svr"][
                            "estimate"
                        ],
                        measure_df[measure_df["regression"] == "mlp"][
                            "estimate"
                        ],
                    )[0],
                    "pair": "mlp-svr",
                    "measure": measure,
                },
                index=[0],
            ),
        )
    ).reset_index(drop=True)
    correlation_df = pd.concat(
        (
            correlation_df,
            pd.DataFrame(
                {
                    "correlation": spearmanr(
                        measure_df[measure_df["regression"] == "rf"][
                            "estimate"
                        ],
                        measure_df[measure_df["regression"] == "svr"][
                            "estimate"
                        ],
                    )[0],
                    "pair": "rf-svr",
                    "measure": measure,
                },
                index=[0],
            ),
        )
    ).reset_index(drop=True)


correlation_df_sig = pd.DataFrame()
correlation_measures = ["CFI_mult", "CFI_unit", "CKE", "NP-CKE"]
for measure in correlation_measures:
    measure_df = (
        res_df[
            (res_df["measure"] == measure) & (res_df["regression"] != "dummy")
        ]
        .sort_values("j")
        .copy()
    )
    measure_df["one_significant"] = (
        measure_df.groupby("j")["significant"].transform("sum") > 0
    )
    measure_df = measure_df[measure_df["one_significant"]]
    correlation_df_sig = pd.concat(
        (
            correlation_df_sig,
            pd.DataFrame(
                {
                    "correlation": spearmanr(
                        measure_df[measure_df["regression"] == "rf"][
                            "scaled_estimate"
                        ],
                        measure_df[measure_df["regression"] == "mlp"][
                            "scaled_estimate"
                        ],
                    )[0],
                    "pair": "rf-mlp",
                    "measure": measure,
                },
                index=[0],
            ),
        )
    ).reset_index(drop=True)
    correlation_df_sig = pd.concat(
        (
            correlation_df_sig,
            pd.DataFrame(
                {
                    "correlation": spearmanr(
                        measure_df[measure_df["regression"] == "svr"][
                            "scaled_estimate"
                        ],
                        measure_df[measure_df["regression"] == "mlp"][
                            "scaled_estimate"
                        ],
                    )[0],
                    "pair": "mlp-svr",
                    "measure": measure,
                },
                index=[0],
            ),
        )
    ).reset_index(drop=True)
    correlation_df_sig = pd.concat(
        (
            correlation_df_sig,
            pd.DataFrame(
                {
                    "correlation": spearmanr(
                        measure_df[measure_df["regression"] == "rf"][
                            "scaled_estimate"
                        ],
                        measure_df[measure_df["regression"] == "svr"][
                            "scaled_estimate"
                        ],
                    )[0],
                    "pair": "rf-svr",
                    "measure": measure,
                },
                index=[0],
            ),
        )
    ).reset_index(drop=True)


fig, axn = plt.subplots(1, 2, figsize=(10, 3))
fig.subplots_adjust(wspace=0.3)
cbar_ax = fig.add_axes([0.96, 0.3, 0.03, 0.4])
cbar_ax.set_title("Spearman\ncorrelation", y=1.05)

for j, sig in enumerate([False, True]):
    ax = axn.flat[j]

    if sig:
        tile_df = correlation_df_sig.pivot(
            index="pair", columns="measure", values="correlation"
        )
    else:
        tile_df = correlation_df.pivot(
            index="pair", columns="measure", values="correlation"
        )

    sns.heatmap(
        tile_df,
        ax=ax,
        annot=True,
        vmax=1,
        vmin=0,
        cbar=(j == 0),
        cbar_ax=None if (j > 0) else cbar_ax,
        cmap=colormap,
    )
    ax.tick_params(axis="y", rotation=0)
    if sig:
        ax.set_title("Correlation on significant species")
        ax.set_xticks(
            np.arange(0, 4) + 0.5,
            [
                r"$\mathrm{CFI}_{\mathrm{mult}}$",
                r"$\mathrm{CFI}_{\mathrm{unit}}$",
                r"$\mathrm{CKE}$",
                r"NP-$\mathrm{CKE}$",
            ],
            rotation=50,
            ha="right",
            rotation_mode="anchor",
        )
    else:
        ax.set_title("Correlation on all species")
        ax.set_xticks(
            np.arange(0, 6) + 0.5,
            [
                r"$\mathrm{CFI}_{\mathrm{mult}}$",
                r"$\mathrm{CFI}_{\mathrm{unit}}$",
                r"$\mathrm{CKE}$",
                r"NP-$\mathrm{CKE}$",
                r"$R^2$",
                r"perm",
            ],
            rotation=50,
            ha="right",
            rotation_mode="anchor",
        )
    ax.set_xlabel("")
    ax.set_ylabel("")
fig.savefig("plots/microbiome_correlation.pdf", bbox_inches="tight")


#### CFIs, log-contrast comparison + main text computations

res_df = pd.read_pickle(
    "experiments/clean_results/microbiome-comparison.pkl"
)
simple_df = pd.read_pickle(
    "experiments/microbiome-simple.pkl"
)
simple_df = simple_df[simple_df["measure"] != "classo"]
cfi_df = pd.concat(
    (
        res_df[
            (res_df["regression"] == "rf")
            & (res_df["measure"].isin(["CFI_mult", "CFI_unit", "CKE"]))
        ],
        simple_df,
    )
).reset_index(drop=True)
cfi_df["variance"] = cfi_df["result"].apply(lambda x: x["variance"])
cfi_df["estimate"] = cfi_df["result"].apply(lambda x: x["estimate"])
cfi_df["standard_error"] = cfi_df["result"].apply(lambda x: x["standard_error"])
cfi_df["p_value"] = 2 * norm.cdf(
    -np.abs(cfi_df["estimate"] / cfi_df["standard_error"])
)
cfi_df["significant"] = cfi_df["p_value"] < 0.05
cfi_df = cfi_df.drop(["result", "seed"], axis=1)

comparison_measures = ["CFI_mult", "CFI_unit", "CKE", "log_contrast"]
estimates = (
    cfi_df[cfi_df["measure"].isin(comparison_measures)]
    .pivot_table(index="j", columns="measure", values="estimate")
    .dropna()
)
print("Spearman rank correlations, {} components:".format(estimates.shape[0]))
for a_idx, a in enumerate(comparison_measures):
    for b in comparison_measures[(a_idx + 1) :]:
        print(
            "{:<12s} vs {:<12s} {:+.2f}".format(
                a, b, spearmanr(estimates[a], estimates[b])[0]
            )
        )
print(
    "median |CFI_unit| / median |CFI_mult| = {:.0f}".format(
        estimates["CFI_unit"].abs().median()
        / estimates["CFI_mult"].abs().median()
    )
)


print("median |estimate| by approach (Y is BMI, so kg/m^2 per unit of L):")
for measure in comparison_measures:
    print(
        "   {:<13s} {:.4g}".format(measure, estimates[measure].abs().median())
    )


L_sd_unit = np.sqrt(
    res_df[(res_df["measure"] == "CFI_unit") & (res_df["regression"] == "rf")][
        "L_var"
    ].astype(float)
)
print(
    "one unit of CFI_unit = +0.5 in z^j = {:.0f} sd of the median "
    "component's abundance (median sd(z^j) = {:.3g})".format(
        1 / np.median(L_sd_unit), np.median(L_sd_unit) / 2
    )
)


var_names = {
    row["j"]: row["var_name"]
    for _, row in simple_df[simple_df["regression"] == "OLS"][
        ["var_name", "j"]
    ].iterrows()
}

custom_var_names = {
    339: "Ruminococcaceae\n1360-2373",
    217: "Lachnospiraceae\n1470-2573",
    248: "Lachnospiraceae\n2819-4963",
    541: "Haemophilus\nparainfluenzae",
    327: "Ruminococcaceae\n1106-1928",
    325: "Ruminococcaceae\n1096-1899",
    222: "Lachnospiraceae\n165-302",
    210: "Lachnospiraceae\n1272-2215",
    476: "Catenibacterium\n1380",
    139: "Turicibacter\n69",
    416: "Ruminococcaceae\n951-1658",
    144: "Clostridiales\n152-578-1023",
    441: "Ruminococcus\n2082",
    308: "Lachnospira\n3094",
    434: "Oscillospira\n4383",
    14: "Bifidobacterium\nanimalis",
    466: "Finegoldia\n293",
    85: "Prevotella\n3926",
    138: "Turicibacter\n4424",
    3: "Corynebacterium\n1249",
    112: "Butyricimonas\n2623",
    133: "Lactococcus\n1275",
    146: "Clostridiales\n173-691-1218",
    86: "Prevotella\n4498",
    505: "Enterobacteriaceae\n1168-2037",
    151: "Clostridiales\n226-911-1597",
    517: "Enterobacteriaceae\n2824-4969",
    552: "Pseudomonas\nveronii",
    62: "Bacteroides\n911",
    8: "Corynebacterium\nsimulans",
    408: "Ruminococcaceae\n819-1452",
    526: "Enterobacteriaceae\n607-1063",
    530: "Enterobacter\n4159",
    514: "Enterobacteriaceae\n2180-3815",
    11: "Bifidobacterium\n1881",
    349: "Ruminococcaceae\n1612-2819",
    131: "Lactobacillus\n2195",
    174: "Clostridiales\n495-1946-3394",
    366: "Ruminococcaceae\n214-379",
    384: "Ruminococcaceae\n2815-4956",
    440: "Ruminococcus\n1862",
    58: "Bacteroides\n488",
    472: "Tissierellaceae\nph2-3202",
    421: "Oscillospira\n1716",
}

n_significant = 10

### Full plot

fig, axn = plt.subplots(2, 2, figsize=(15, 8))
fig.subplots_adjust(hspace=0.55, top=0.875, wspace=0.25)

color_dict = {
    "CFI_mult": color1,  # blue
    "log_contrast": color2,  # green
    "OLS": color3,  # red,
    "CKE": color4,
}

rf_p = cfi_df[cfi_df["measure"] == "CFI_mult"].sort_values("p_value")[
    ["j", "p_value"]
]
rf_significant_j = rf_p["j"].iloc[0:n_significant].to_numpy()


p_colormap = truncate_colormap(
    sns.color_palette("rocket", as_cmap=True), 0.2, 0.9
)
rf_p_colors = {
    int(x["j"]): p_colormap(np.log10(x["p_value"]) / 6 + 4 / 3)
    for _, x in rf_p.iterrows()
}

cfi_df_rf = cfi_df[cfi_df["j"].isin(rf_significant_j)].copy()
cfi_df_rf["j"] = pd.Categorical(
    cfi_df_rf["j"], rf_significant_j, ordered=True
)  # permits custom sorting
cfi_df_rf = cfi_df_rf.sort_values("j")

ax = axn[0, 0]
for j, measure in enumerate(["CFI_mult", "log_contrast", "OLS"]):
    sub_df = cfi_df_rf[cfi_df_rf["measure"] == measure]
    ax.errorbar(
        -0.15 + 0.15 * j + np.arange(n_significant),
        sub_df["estimate"],
        yerr=1.96 * sub_df["standard_error"],
        marker="o",
        linestyle="none",
        c=color_dict[measure],
        alpha=0.8,
        linewidth=1.5,
    )
    ax.set_xticks(
        ticks=np.arange(0, n_significant),
        labels=[custom_var_names[j] for j in rf_significant_j],
        rotation=50,
        fontsize=10,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_ylim(-0.4, 0.4)
    ax.set_title(
        r"Top $\mathrm{CFI}_{\mathrm{mult}}$ significant species", pad=10
    )
    ax.set_ylabel("Effect estimate")
for i, xtick in enumerate(ax.get_xticklabels()):
    xtick.set_color(rf_p_colors[rf_significant_j[i]])


log_contrast_p = cfi_df[cfi_df["measure"] == "log_contrast"].sort_values(
    "p_value"
)[["j", "p_value"]]
log_contrast_significant_j = (
    log_contrast_p["j"].iloc[0:n_significant].to_numpy()
)

log_contrast_p_colors = {
    int(x["j"]): p_colormap(np.log10(x["p_value"]) / 6 + 4 / 3)
    for _, x in log_contrast_p.iterrows()
}

cfi_df_log_contrast = cfi_df[
    cfi_df["j"].isin(log_contrast_significant_j)
].copy()
cfi_df_log_contrast["j"] = pd.Categorical(
    cfi_df_log_contrast["j"], log_contrast_significant_j, ordered=True
)  # permits custom sorting
cfi_df_log_contrast = cfi_df_log_contrast.sort_values("j")


ax = axn[0, 1]
for j, measure in enumerate(["CFI_mult", "log_contrast", "OLS"]):
    sub_df = cfi_df_log_contrast[cfi_df_log_contrast["measure"] == measure]
    ax.errorbar(
        -0.15 + 0.15 * j + np.arange(n_significant),
        sub_df["estimate"],
        yerr=1.96 * sub_df["standard_error"],
        marker="o",
        linestyle="none",
        c=color_dict[measure],
        alpha=0.8,
        linewidth=1.5,
    )
    ax.set_xticks(
        ticks=np.arange(0, n_significant),
        labels=[custom_var_names[j] for j in log_contrast_significant_j],
        rotation=50,
        fontsize=10,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_ylim(-0.2, 0.2)
    ax.set_title(r"Top log-contrast significant species", pad=10)
    ax.set_ylabel("")

for i, xtick in enumerate(ax.get_xticklabels()):
    xtick.set_color(log_contrast_p_colors[log_contrast_significant_j[i]])


ax = axn[1, 0]
significant_j = list(
    cfi_df.iloc[idx]["j"]
    for _, idx in cfi_df[cfi_df["measure"] != "OLS"]
    .groupby("measure")["p_value"]
    .nsmallest(3)
    .index
)
significant_j = list(dict.fromkeys(significant_j))  # removes duplicates

tile_df = (
    cfi_df[(cfi_df["measure"] != "OLS") & (cfi_df["j"].isin(significant_j))]
    .pivot(index="measure", columns="j", values="p_value")
    .reindex(significant_j, axis=1)
)
tile_df = 10.0 ** np.floor((np.log10(tile_df))).astype(int)

sns.heatmap(
    tile_df,
    ax=ax,
    annot=True,
    cmap=p_colormap,
    norm=colors.LogNorm(vmin=10 ** (-8), vmax=10 ** (-2)),
    cbar=False,
)
for t in ax.texts:
    t.set_text(
        r"$10^{{ {x} }}$".format(
            x=np.floor((np.log10(float(t.get_text())))).astype(int)
        )
    )
ax.set_ylabel("")
ax.set_xlabel("")
ax.set_xticks(
    ticks=np.arange(0, len(significant_j)) + 0.5,
    labels=[custom_var_names[j] for j in significant_j],
    rotation=50,
    fontsize=10,
    ha="right",
    rotation_mode="anchor",
)
ax.set_yticks(
    ticks=np.arange(0, 4) + 0.5,
    labels=[
        r"$\mathrm{CFI}_{\mathrm{mult}}$",
        r"$\mathrm{CFI}_{\mathrm{unit}}$",
        r"$\mathrm{CKE}$",
        r"log-contrast",
    ],
)
ax.set_title(r"$p$-values for most significant species")

ax = axn[1, 1]
log_contrast_df = pd.read_pickle(
    "experiments/microbiome-pseudocount.pkl"
)
log_contrast_df["estimate"] = log_contrast_df["result"].apply(
    lambda x: x["estimate"]
)
log_contrast_df["scaled_estimate"] = log_contrast_df["estimate"] * np.sqrt(
    log_contrast_df["L_var"].astype(float)
)
log_contrast_df = log_contrast_df[
    log_contrast_df["j"].isin(log_contrast_significant_j)
]
log_contrast_df = log_contrast_df[
    (np.log10(log_contrast_df["pseudo_count"]) < -1)
    & (np.log10(log_contrast_df["pseudo_count"]) > -8)
]
g = sns.lineplot(
    x="pseudo_count",
    y="scaled_estimate",
    style="j",
    data=log_contrast_df,
    ax=ax,
    color="black",
    legend=False,
)
g.set(xscale="log", ylim=(-0.5, 0.7))
ax.axvline(x=9.509391475020732e-07, linestyle="dashed", c="lightgrey")
ax.set_xlabel("Pseudocount")
ax.set_ylabel("Log-contrast scaled effect")
ax.set_title(r"Top log-contrast scaled effects")


normalizer = plt.Normalize(-8, -2)
sm = ScalarMappable(norm=normalizer, cmap=p_colormap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axn, aspect=50, pad=0.025)
cbar.ax.set_title(r"$p$")
cbar.set_ticks(
    ticks=[-8, -6, -4, -2],
    labels=[r"$10^{-8}$", r"$10^{-6}$", r"$10^{-4}$", r"$10^{-2}$"],
)


legend_elements = [
    Line2D(
        [0], [0], marker="o", color="white", markerfacecolor=col, markersize=10
    )
    for col in color_dict.values()
]

fig.legend(
    legend_elements,
    [
        r"$\mathrm{CFI}_{\mathrm{mult}}$",
        r"log-contrast",
        r"Marginal $L$ effect",
    ],
    loc="upper center",
    ncol=4,
)
fig.savefig("plots/microbiome_cfi_comparison.pdf", bbox_inches="tight")


#### plot 1/2
fig, axn = plt.subplots(1, 2, figsize=(15, 4))
fig.subplots_adjust(top=0.775, wspace=0.15)

color_dict = {
    "CFI_mult": color1,  # blue
    "log_contrast": color2,  # green
    "OLS": color3,  # red,
    "CKE": color4,
}

rf_p = cfi_df[cfi_df["measure"] == "CFI_mult"].sort_values("p_value")[
    ["j", "p_value"]
]
rf_significant_j = rf_p["j"].iloc[0:n_significant].to_numpy()


p_colormap = truncate_colormap(
    sns.color_palette("rocket", as_cmap=True), 0.2, 0.9
)
rf_p_colors = {
    int(x["j"]): p_colormap(np.log10(x["p_value"]) / 6 + 4 / 3)
    for _, x in rf_p.iterrows()
}

cfi_df_rf = cfi_df[cfi_df["j"].isin(rf_significant_j)].copy()
cfi_df_rf["j"] = pd.Categorical(
    cfi_df_rf["j"], rf_significant_j, ordered=True
)  # permits custom sorting
cfi_df_rf = cfi_df_rf.sort_values("j")

ax = axn[0]
for j, measure in enumerate(["CFI_mult", "log_contrast", "OLS"]):
    sub_df = cfi_df_rf[cfi_df_rf["measure"] == measure]
    ax.errorbar(
        -0.15 + 0.15 * j + np.arange(n_significant),
        sub_df["estimate"],
        yerr=1.96 * sub_df["standard_error"],
        marker="o",
        linestyle="none",
        c=color_dict[measure],
        alpha=0.8,
        linewidth=1.5,
    )
    ax.set_xticks(
        ticks=np.arange(0, n_significant),
        labels=[custom_var_names[j] for j in rf_significant_j],
        rotation=50,
        fontsize=11,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_ylim(-0.4, 0.4)
    ax.set_title(
        r"Top $\mathrm{CFI}_{\mathrm{mult}}$ significant species", pad=10
    )
    ax.set_ylabel("Effect estimate")
for i, xtick in enumerate(ax.get_xticklabels()):
    xtick.set_color(rf_p_colors[rf_significant_j[i]])


log_contrast_p = cfi_df[cfi_df["measure"] == "log_contrast"].sort_values(
    "p_value"
)[["j", "p_value"]]
log_contrast_significant_j = (
    log_contrast_p["j"].iloc[0:n_significant].to_numpy()
)

log_contrast_p_colors = {
    int(x["j"]): p_colormap(np.log10(x["p_value"]) / 6 + 4 / 3)
    for _, x in log_contrast_p.iterrows()
}

cfi_df_log_contrast = cfi_df[
    cfi_df["j"].isin(log_contrast_significant_j)
].copy()
cfi_df_log_contrast["j"] = pd.Categorical(
    cfi_df_log_contrast["j"], log_contrast_significant_j, ordered=True
)  # permits custom sorting
cfi_df_log_contrast = cfi_df_log_contrast.sort_values("j")


ax = axn[1]
for j, measure in enumerate(["CFI_mult", "log_contrast", "OLS"]):
    sub_df = cfi_df_log_contrast[cfi_df_log_contrast["measure"] == measure]
    ax.errorbar(
        -0.15 + 0.15 * j + np.arange(n_significant),
        sub_df["estimate"],
        yerr=1.96 * sub_df["standard_error"],
        marker="o",
        linestyle="none",
        c=color_dict[measure],
        alpha=0.8,
        linewidth=1.5,
    )
    ax.set_xticks(
        ticks=np.arange(0, n_significant),
        labels=[custom_var_names[j] for j in log_contrast_significant_j],
        rotation=50,
        fontsize=11,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_ylim(-0.2, 0.2)
    ax.set_title(r"Top log-contrast significant species", pad=10)
    ax.set_ylabel("")

for i, xtick in enumerate(ax.get_xticklabels()):
    xtick.set_color(log_contrast_p_colors[log_contrast_significant_j[i]])


normalizer = plt.Normalize(-8, -2)
sm = ScalarMappable(norm=normalizer, cmap=p_colormap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axn, pad=0.025)
cbar.ax.set_title(r"$p$", y=1.02)
cbar.set_ticks(
    ticks=[-8, -6, -4, -2],
    labels=[r"$10^{-8}$", r"$10^{-6}$", r"$10^{-4}$", r"$10^{-2}$"],
)


legend_elements = [
    Line2D(
        [0], [0], marker="o", color="white", markerfacecolor=col, markersize=10
    )
    for col in color_dict.values()
]

fig.legend(
    legend_elements,
    [
        r"$\mathrm{CFI}_{\mathrm{mult}}$",
        r"log-contrast",
        r"Marginal $L$ effect",
    ],
    loc="upper center",
    ncol=4,
)
fig.savefig("plots/microbiome_cfi_comparison1.pdf", bbox_inches="tight")


#### plot 2/2

fig, axn = plt.subplots(1, 2, figsize=(15, 4))
fig.subplots_adjust(wspace=0.25)

color_dict = {
    "CFI_mult": color1,  # blue
    "log_contrast": color2,  # green
    "OLS": color3,  # red,
    "CKE": color4,
}

rf_p = cfi_df[cfi_df["measure"] == "CFI_mult"].sort_values("p_value")[
    ["j", "p_value"]
]
rf_significant_j = rf_p["j"].iloc[0:n_significant].to_numpy()


p_colormap = truncate_colormap(
    sns.color_palette("rocket", as_cmap=True), 0.2, 0.9
)
rf_p_colors = {
    int(x["j"]): p_colormap(np.log10(x["p_value"]) / 6 + 4 / 3)
    for _, x in rf_p.iterrows()
}

cfi_df_rf = cfi_df[cfi_df["j"].isin(rf_significant_j)].copy()
cfi_df_rf["j"] = pd.Categorical(
    cfi_df_rf["j"], rf_significant_j, ordered=True
)  # permits custom sorting
cfi_df_rf = cfi_df_rf.sort_values("j")


ax = axn[0]
significant_j = list(
    cfi_df.iloc[idx]["j"]
    for _, idx in cfi_df[cfi_df["measure"] != "OLS"]
    .groupby("measure")["p_value"]
    .nsmallest(3)
    .index
)
significant_j = list(dict.fromkeys(significant_j))  # removes duplicates

tile_df = (
    cfi_df[(cfi_df["measure"] != "OLS") & (cfi_df["j"].isin(significant_j))]
    .pivot(index="measure", columns="j", values="p_value")
    .reindex(significant_j, axis=1)
)
tile_df = 10.0 ** np.floor((np.log10(tile_df))).astype(int)

sns.heatmap(
    tile_df,
    ax=ax,
    annot=True,
    cmap=p_colormap,
    norm=colors.LogNorm(vmin=10 ** (-8), vmax=10 ** (-2)),
    cbar=False,
)
for t in ax.texts:
    t.set_text(
        r"$10^{{ {x} }}$".format(
            x=np.floor((np.log10(float(t.get_text())))).astype(int)
        )
    )
ax.set_ylabel("")
ax.set_xlabel("")
ax.set_xticks(
    ticks=np.arange(0, len(significant_j)) + 0.5,
    labels=[custom_var_names[j] for j in significant_j],
    rotation=50,
    fontsize=11,
    ha="right",
    rotation_mode="anchor",
)
ax.set_yticks(
    ticks=np.arange(0, 4) + 0.5,
    labels=[
        r"$\mathrm{CFI}_{\mathrm{mult}}$",
        r"$\mathrm{CFI}_{\mathrm{unit}}$",
        r"$\mathrm{CKE}$",
        r"log-contrast",
    ],
)
ax.set_title(r"$p$-values for most significant species")

ax = axn[1]
log_contrast_df = pd.read_pickle(
    "experiments/microbiome-pseudocount.pkl"
)
log_contrast_df["estimate"] = log_contrast_df["result"].apply(
    lambda x: x["estimate"]
)
log_contrast_df["scaled_estimate"] = log_contrast_df["estimate"] * np.sqrt(
    log_contrast_df["L_var"].astype(float)
)
log_contrast_df = log_contrast_df[
    log_contrast_df["j"].isin(log_contrast_significant_j)
]
log_contrast_df = log_contrast_df[
    (np.log10(log_contrast_df["pseudo_count"]) < -1)
    & (np.log10(log_contrast_df["pseudo_count"]) > -8)
]
g = sns.lineplot(
    x="pseudo_count",
    y="scaled_estimate",
    style="j",
    data=log_contrast_df,
    ax=ax,
    color="black",
    legend=False,
)
g.set(xscale="log", ylim=(-0.5, 0.7))
ax.axvline(x=9.509391475020732e-07, linestyle="dashed", c="lightgrey")
ax.set_xlabel("Pseudocount")
ax.set_ylabel("Log-contrast scaled effect")
ax.set_title(r"Top log-contrast scaled effects")

normalizer = plt.Normalize(-8, -2)
sm = ScalarMappable(norm=normalizer, cmap=p_colormap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axn, pad=0.025)
cbar.ax.set_title(r"$p$", y=1.02)
cbar.set_ticks(
    ticks=[-8, -6, -4, -2],
    labels=[r"$10^{-8}$", r"$10^{-6}$", r"$10^{-4}$", r"$10^{-2}$"],
)


fig.savefig("plots/microbiome_cfi_comparison2.pdf", bbox_inches="tight")


### CFIs CKE comparison


res_df = pd.read_pickle(
    "experiments/clean_results/microbiome-comparison.pkl"
)
cfi_df = res_df[
    (res_df["measure"].isin(["CKE", "CFI_mult", "CFI_unit"]))
    & (res_df["regression"] == "rf")
].copy()
cfi_df["variance"] = cfi_df["result"].apply(lambda x: x["variance"])
cfi_df["estimate"] = cfi_df["result"].apply(lambda x: x["estimate"])
cfi_df.loc[cfi_df["measure"] == "CKE", "estimate"] = -cfi_df.loc[
    cfi_df["measure"] == "CKE", "estimate"
]
cfi_df["standard_error"] = cfi_df["result"].apply(lambda x: x["standard_error"])
cfi_df["p_value"] = 2 * norm.cdf(
    -np.abs(cfi_df["estimate"] / cfi_df["standard_error"])
)
cfi_df["significant"] = cfi_df["p_value"] < 0.05
cfi_df = cfi_df.drop(["result", "seed"], axis=1)
cfi_df["scaled_estimate"] = cfi_df["estimate"] * np.sqrt(
    cfi_df["L_var"].astype(float)
)
cfi_df["scaled_standard_error"] = cfi_df["standard_error"] * np.sqrt(
    cfi_df["L_var"].astype(float)
)

n_significant = 10

fig, axn = plt.subplots(1, 2, figsize=(15, 4))
fig.subplots_adjust(top=0.775)

color_dict = {
    "CFI_mult": color1,  # blue
    "CFI_unit": color4,
    "CKE": color5,
}

unit_p = cfi_df[cfi_df["measure"] == "CFI_unit"].sort_values("p_value")[
    ["j", "p_value"]
]
unit_significant_j = unit_p["j"].iloc[0:n_significant].to_numpy()


unit_p_colors = {
    int(x["j"]): p_colormap(np.log10(x["p_value"]) / 6 + 4 / 3)
    for _, x in unit_p.iterrows()
}

cfi_df_unit = cfi_df[cfi_df["j"].isin(unit_significant_j)].copy()
cfi_df_unit["j"] = pd.Categorical(
    cfi_df_unit["j"], unit_significant_j, ordered=True
)  # permits custom sorting
cfi_df_unit = cfi_df_unit.sort_values("j")

ax = axn[0]
for j, measure in enumerate(["CFI_mult", "CFI_unit", "CKE"]):
    sub_df = cfi_df_unit[cfi_df_unit["measure"] == measure]
    ax.errorbar(
        -0.15 + 0.15 * j + np.arange(n_significant),
        sub_df["scaled_estimate"],
        yerr=1.96 * sub_df["scaled_standard_error"],
        marker="o",
        linestyle="none",
        c=color_dict[measure],
        alpha=0.8,
        linewidth=1.5,
    )
    ax.set_xticks(
        ticks=np.arange(0, n_significant),
        labels=[custom_var_names[j] for j in unit_significant_j],
        rotation=50,
        fontsize=11,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_ylim(-0.6, 0.5)
    ax.set_title(
        r"Top $\mathrm{CFI}_{\mathrm{unit}}$ significant species", pad=10
    )
    ax.set_ylabel("Scaled effect estimate")
for i, xtick in enumerate(ax.get_xticklabels()):
    xtick.set_color(unit_p_colors[unit_significant_j[i]])


cke_p = cfi_df[cfi_df["measure"] == "CKE"].sort_values("p_value")[
    ["j", "p_value"]
]
cke_significant_j = cke_p["j"].iloc[0:n_significant].to_numpy()

cke_p_colors = {
    int(x["j"]): p_colormap(np.log10(x["p_value"]) / 6 + 4 / 3)
    for _, x in cke_p.iterrows()
}

cfi_df_cke = cfi_df[cfi_df["j"].isin(cke_significant_j)].copy()
cfi_df_cke["j"] = pd.Categorical(
    cfi_df_cke["j"], cke_significant_j, ordered=True
)  # permits custom sorting
cfi_df_cke = cfi_df_cke.sort_values("j")


ax = axn[1]
for j, measure in enumerate(["CFI_mult", "CFI_unit", "CKE"]):
    sub_df = cfi_df_cke[cfi_df_cke["measure"] == measure]
    ax.errorbar(
        -0.15 + 0.15 * j + np.arange(n_significant),
        sub_df["scaled_estimate"],
        yerr=1.96 * sub_df["scaled_standard_error"],
        marker="o",
        linestyle="none",
        c=color_dict[measure],
        alpha=0.8,
        linewidth=1.5,
    )
    ax.set_xticks(
        ticks=np.arange(0, n_significant),
        labels=[custom_var_names[j] for j in cke_significant_j],
        rotation=50,
        fontsize=11,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_ylim(-0.6, 0.7)
    ax.set_title(r"Top CKE significant species", pad=10)

for i, xtick in enumerate(ax.get_xticklabels()):
    xtick.set_color(cke_p_colors[cke_significant_j[i]])

normalizer = plt.Normalize(-8, -2)
sm = ScalarMappable(norm=normalizer, cmap=p_colormap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axn, pad=0.025)
cbar.ax.set_title(r"$p$", y=1.02)
cbar.set_ticks(
    ticks=[-8, -6, -4, -2],
    labels=[r"$10^{-8}$", r"$10^{-6}$", r"$10^{-4}$", r"$10^{-2}$"],
)


legend_elements = [
    Line2D(
        [0], [0], marker="o", color="white", markerfacecolor=col, markersize=10
    )
    for col in color_dict.values()
]

fig.legend(
    legend_elements,
    [
        r"$\mathrm{CFI}_{\mathrm{mult}}$",
        r"$\mathrm{CFI}_{\mathrm{unit}}$",
        r"$\mathrm{CKE}$",
    ],
    loc="upper center",
    ncol=4,
)
fig.savefig("plots/microbiome_cke_comparison.pdf", bbox_inches="tight")


### PLM comparison

res_df = pd.read_pickle("experiments/clean_results/microbiome-dml_15-11-23.pkl")
res_df["variance"] = res_df["result"].apply(lambda x: x["variance"])
res_df["estimate"] = res_df["result"].apply(lambda x: x["estimate"])
res_df["standard_error"] = res_df["result"].apply(lambda x: x["standard_error"])
res_df["p_value"] = 2 * norm.cdf(
    -np.abs(res_df["estimate"] / res_df["standard_error"])
)
res_df["scaled_estimate"] = (
    np.sqrt(res_df["L_var"].astype(float)) * res_df["estimate"]
)
res_df["absolute_scaled_estimate"] = res_df["scaled_estimate"].abs()
res_df["squared_scaled_estimate"] = res_df["scaled_estimate"] ** 2
res_df["var_explained"] = res_df["squared_scaled_estimate"] / res_df["Y_var"]
res_df["significant"] = res_df["p_value"] < 0.05
res_df = res_df.drop(["result", "seed"], axis=1)


print(
    "variance explained, var(L) * estimate^2 / var(Y), by species "
    "(Section S5.7):"
)
for measure, label in (("CFI_unit", "CFI_unit"), ("DML", "naive linear")):
    var_explained = res_df.loc[
        res_df["measure"] == measure, "var_explained"
    ].astype(float)
    print(
        "   {:<13s} {:d} species, from {:.3g} to {:.3g} "
        "(log10: {:.2f} to {:.2f})".format(
            label,
            var_explained.size,
            var_explained.min(),
            var_explained.max(),
            np.log10(var_explained.min()),
            np.log10(var_explained.max()),
        )
    )
