import pandas as pd
import numpy as np
import seaborn as sns 
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.lines import Line2D
import matplotlib as mpl

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = colors.LinearSegmentedColormap.from_list(
         'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
         cmap(np.linspace(minval, maxval, n)))
    return new_cmap

colormap = truncate_colormap(mpl.colormaps["viridis"], 0.2, 0.9)

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 16})
plt.rc('text.latex', preamble=r'\usepackage{amsfonts,amssymb,amsthm,amsmath}')
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

df = pd.read_pickle("experiments/clean_results/semiparametric.pkl")

df["variance"] = df["result"].apply(lambda x:x["variance"])
df["estimate"] = df["result"].apply(lambda x:x["estimate"])
df["standard_error"] = df["result"].apply(lambda x:x["standard_error"])
df["z"] = (df["estimate"]-1)/df["standard_error"]
df = df.rename(columns={"p": "d"})
df["covered"] = np.abs(df["z"]) <= norm.ppf(0.975)

plot_df = df.groupby(["n", "d", "Y_regression", "estimator", "type"])["covered"].mean().reset_index(name="coverage")

for d in plot_df["d"].unique():
    tile_df = plot_df[plot_df["d"] == d]
    tile_df = tile_df.drop("d", axis=1)
    tile_df["estimator"] = tile_df["estimator"].replace(
        {"partially_linear":"PLM", "partially_linear_no_x":"PLM_no_x",
         "onestep": "NPM",
         "onestep_true": "NPM_oracle",
         "onestep_no_x":"NPM_no_x"})
    fig, axn = plt.subplots(2, 2, figsize=(10, 7))
    fig.subplots_adjust(hspace=0.55, wspace=0.1)
    cbar_ax = fig.add_axes([.95, .3, 0.03, .4])
    cbar_ax.set_title("Coverage", y=1.05)
    
    for j, type in enumerate(tile_df["type"].unique()):
        for i, y_reg in enumerate(tile_df["Y_regression"].unique()):
            ax = axn.flat[i+2*j]
            
            subtile_df = tile_df[(tile_df["Y_regression"] == y_reg) & (tile_df["type"] == type)].pivot(index="estimator", columns="n", values="coverage")
            sns.heatmap(subtile_df, ax=ax, annot=True, vmax=1, vmin=0, cbar=(i==0) & (j==0), cbar_ax = None if (i > 0) or (j>0) else cbar_ax, cmap=colormap)
            ax.set_title("{} $L$\n{} $Y$".format(type.capitalize(), y_reg.capitalize().replace("_", " ")))
            ax.set_xlabel("$n$")
            ax.set_ylabel("")
            if i != 0:
                ax.set_yticks([])
            else:
                ax.set_yticks(np.arange(0, 7) + 0.5, [r"$\texttt{NPM}$", r"$\texttt{NPM\_no\_x}$", r"$\texttt{NPM\_oracle}$", r"$\texttt{PLM}$", r"$\texttt{PLM\_no\_x}$", r"$\texttt{plugin}$", r"$\texttt{plugin\_no\_x}$"])
    fig.savefig("plots/semiparametric_{}.pdf".format(d), bbox_inches="tight")


fig, axn = plt.subplots(2, 3, figsize=(10, 6))
fig.subplots_adjust(wspace=0.15, top=0.8)
plot_df = df[(df["d"] == 15) & (df["Y_regression"] == "nonparametric") & (df["estimator"].isin(["onestep", "partially_linear"]))]

for j, n in enumerate([250, 1000, 4000]):
    for i, typ in enumerate(["binary", "continuous"]):
        sub_df = plot_df[(plot_df["type"] == typ) & (plot_df["n"] == n)]
        sns.kdeplot(sub_df, x="estimate", ax=axn[i, j], hue="estimator", legend=False, palette={"partially_linear": "C2", "onestep": "C1"})
        if i == 0:
            axn[i, j].set_title(r"$n={}$".format(n))
            axn[i, j].set_xlim(0.6, 1.2)
            axn[i, j].set_ylim(0, 5)
            axn[i, j].set_xlabel("")
            if j == 0:
                axn[i, j].set_yticks([0, 2, 4])
        else:
            axn[i, j].set_xlim(0.8, 1.8)
            axn[i, j].set_ylim(0, 10)
            axn[i, j].set_xlabel("Estimate")
        if j == 0:
            axn[i, j].set_ylabel(r"{} $L$".format(typ.capitalize()) + "\n\n Density")
            if i == 1:
                axn[i, j].set_yticks([0, 4, 8])
        else:
            axn[i, j].set_ylabel("")
            axn[i, j].set_yticks([])


fig.suptitle(r"Distribution of estimates with $d=15$", y=0.9)

legend_elements = [
    Line2D([0], [0], marker="o", color="white", markerfacecolor="C1", markersize=10),
    Line2D([0], [0], marker="o", color="white", markerfacecolor="C2", markersize=10)]

fig.legend(legend_elements, ["NPM", "PLM"], loc="upper center", ncol=2)

fig.savefig("plots/semiparametric_distributions.pdf", bbox_inches="tight")
