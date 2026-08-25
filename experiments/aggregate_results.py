"""Aggregate per-run simulation results into the tables read by the plot scripts.

`experiments/sim_semiparametric.py` and `experiments/sim_microbiome.py` each run
a single setting of a simulation study per invocation and write one result file
into `experiments/`.  This script collects those files into the combined tables
that the scripts in `plot_scripts/` expect.

Run it from the repository root, after the individual runs have finished:

    PYTHONPATH=. python3 experiments/aggregate_results.py

It writes into `experiments/clean_results/` whichever of the following it finds
inputs for, and reports what it wrote:

    semiparametric.pkl          all runs of sim_semiparametric.py
    microbiome-comparison.pkl   runs of sim_microbiome.py using the rf, mlp, svr
                                and dummy regressions (Figures 2, S11, S12, S13)
    microbiome-dml.pkl          runs of sim_microbiome.py using the
                                cross-validated regression

The two tables written directly by `sim_microbiome_simple.py`
(`experiments/microbiome-simple.pkl` and `experiments/microbiome-pseudocount.pkl`)
are already in their final form and are read from there by the plot scripts, so
they are not touched here.
"""

import glob
import os

import pandas as pd

RESULTS_DIR = "experiments"
CLEAN_DIR = os.path.join(RESULTS_DIR, "clean_results")

# Measures written by sim_microbiome.py as one file per species. "perm" is
# handled separately below because it scores all species in a single run.
PER_SPECIES_MEASURES = ["CKE", "NP-CKE", "CFI_unit", "CFI_mult", "DML", "R2"]

# The regressions used for the comparison figures, and the cross-validated
# regression, are aggregated into separate tables.
COMPARISON_REGRESSIONS = ["rf", "mlp", "svr", "dummy"]
DML_REGRESSIONS = ["cv"]

# sim_semiparametric.py records the estimator under the display names used in
# the paper, and the predictor dimension as "d". The plot scripts use the
# internal names and call the dimension "p".
ESTIMATOR_NAMES = {
    "NPM": "onestep",
    "NPM_no_x": "onestep_no_x",
    "NPM_oracle": "onestep_true",
    "PLM": "partially_linear",
    "PLM_no_x": "partially_linear_no_x",
    "plugin": "plugin",
    "plugin_no_x": "plugin_no_x",
}


def aggregate_semiparametric():
    """Collect the runs of sim_semiparametric.py into a single table."""
    paths = sorted(glob.glob(os.path.join(RESULTS_DIR, "semiparametrics-*.pkl")))
    if not paths:
        return None
    df = pd.DataFrame([pd.read_pickle(path) for path in paths])
    df = df.rename(columns={"d": "p"})
    df["estimator"] = df["estimator"].map(ESTIMATOR_NAMES).fillna(df["estimator"])
    return df.reset_index(drop=True)


def aggregate_microbiome(regressions):
    """Collect the runs of sim_microbiome.py that used the given regressions."""
    frames = []
    for regression in regressions:
        for measure in PER_SPECIES_MEASURES:
            pattern = "{measure}-{regression}-*.pkl".format(
                measure=measure, regression=regression
            )
            paths = sorted(glob.glob(os.path.join(RESULTS_DIR, pattern)))
            if paths:
                frames.append(pd.DataFrame([pd.read_pickle(p) for p in paths]))
        # "perm" writes a whole table rather than one row per species.
        perm_path = os.path.join(
            RESULTS_DIR, "perm-{regression}.pkl".format(regression=regression)
        )
        if os.path.exists(perm_path):
            frames.append(pd.read_pickle(perm_path))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def write(df, name):
    if df is None:
        print("no inputs found for {name}.pkl, skipping".format(name=name))
        return
    path = os.path.join(CLEAN_DIR, name + ".pkl")
    df.to_pickle(path)
    print("wrote {path} ({rows} rows)".format(path=path, rows=df.shape[0]))


if __name__ == "__main__":
    os.makedirs(CLEAN_DIR, exist_ok=True)
    write(aggregate_semiparametric(), "semiparametric")
    write(aggregate_microbiome(COMPARISON_REGRESSIONS), "microbiome-comparison")
    write(aggregate_microbiome(DML_REGRESSIONS), "microbiome-dml")
