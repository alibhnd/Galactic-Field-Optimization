"""
Final engineering-case-study experiment for the GFO manuscript.

Cases
-----
1) Carbonation-depth prediction:
   819 samples, 14 predictors, target = carbonation depth.
2) Recycled geopolymer concrete:
   373 samples, 15 predictors, target = compressive strength.

Model
-----
RBF-SVR. GFO, PSO, DE, GWO, WOA, GA, and Random Search optimize
(log10 C, log10 gamma, log10 epsilon) by minimizing 5-fold CV-RMSE
on the training set only.

IMPORTANT
---------
Run this script from the repository root after installing the package.

That file is imported so the engineering experiment uses exactly the final
continuous GFO and comparator implementations already used in the paper.

Install:
    pip install numpy pandas scipy scikit-learn openpyxl matplotlib

Set CARBONATION_FILE and GEOPOLYMER_FILE below if the workbooks are not in the script folder.
The script accepts .xlsx, .xls, and .csv files and attempts to identify
columns by aliases. If a column cannot be identified, it prints the actual
column names and stops so the mapping can be corrected explicitly.
"""

from __future__ import annotations

import json
import time
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, rankdata
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------------------------------------------------
# Import the FINAL optimizer implementations used in the manuscript
# ---------------------------------------------------------------------
from gfo.continuous import (
    GFO, GFOParams, BaseOptParams,
    pso_opt, de_opt, gwo_opt, woa_opt, ga_opt, random_search_opt
)

# ========================= USER SETTINGS ==============================
CARBONATION_FILE = Path("data") / "Carbonation.xlsx"
GEOPOLYMER_FILE = Path("data") / "Geopolymer.xlsx"

CARBONATION_SHEET = "Carbonation"
GEOPOLYMER_SHEET = "CS_Recycled Geopolymer"

OUTPUT_DIR = Path("results/engineering")

# Reproducible data split
TEST_SIZE = 0.20
DATA_SPLIT_SEED = 42

# Optimization protocol.
# These values are deliberately explicit and common across all algorithms.
N_AGENTS = 20
ITERS = 150
N_RUNS = 10
SEED0 = 42

# Five-fold CV, as described in the manuscript.
N_FOLDS = 5

# Search is performed in log10 space:
# C:       1e-2 ... 1e4
# gamma:   1e-4 ... 1
# epsilon: 1e-3 ... 1
LB = np.array([-2.0, -4.0, -3.0], dtype=float)
UB = np.array([ 4.0,  0.0,  0.0], dtype=float)

ALGORITHMS = ["GFO", "PSO", "DE", "GWO", "WOA", "GA", "RandomSearch"]
# =====================================================================


def norm_name(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())


def read_table(path: str | Path, sheet_name=None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"\nDataset not found: {path}\n"
            f"Edit the file path near the top of this script."
        )
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path, sheet_name=sheet_name)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def find_column(df: pd.DataFrame, aliases: list[str]) -> str:
    normalized = {norm_name(c): c for c in df.columns}
    for a in aliases:
        na = norm_name(a)
        if na in normalized:
            return normalized[na]
    raise KeyError(
        f"Could not identify a required column from aliases: {aliases}\n"
        f"Available columns:\n{list(df.columns)}"
    )


# Exact variable sets described in the current manuscript.
CARBONATION_COLUMNS = {
    "Cement": ["cement", "cement content", "C (kg/m3)"],
    "FlyAsh": ["fly ash", "flyash", "FlA (kg/m3)"],
    "Water": ["water", "water content", "Water (kg/m3)"],
    "WaterBinderRatio": ["water to binder ratio", "water/binder ratio", "w/b", "wb", "wbratio", "W/B"],
    "CoarseAggregate": ["coarse aggregate", "coarseaggregate", "ca", "Gravel (kg/m3)"],
    "RecycledAggregate": ["recycled aggregate", "recycledaggregate", "ra", "rca", "RA (kg/m3)"],
    "EquivalentWaterAbsorption": [
        "equivalent water absorption of coarse aggregate",
        "equivalent water absorption", "ewa", "Equivalent WA of CA (%)"
    ],
    "FineAggregate": ["fine aggregate", "fineaggregate", "sand", "fa aggregate", "FA (kg/m3)"],
    "Superplasticizer": ["superplasticizer", "superplasticizer dosage", "sp", "SP (kg/m3)"],
    "Strength28d": [
        "28-day compressive strength", "28 day compressive strength",
        "compressive strength 28d", "fc28", "strength28", "Cube CS_28 (MPa)"
    ],
    "CO2": [
        "carbonation concentration", "co2 concentration",
        "carbon dioxide concentration", "co2", "Carbon concentration (%)"
    ],
    "ExposureTime": ["exposure time", "carbonation time", "time", "Exposure time (days)"],
    "Temperature": ["temperature", "temp", "Temp (C)"],
    "RelativeHumidity": ["relative humidity", "rh", "RH (%)"],
}
CARBONATION_TARGET = [
    "carbonation depth", "carbonationdepth", "depth", "cd", "Carbonation depth (mm)"
]

GEOPOLYMER_COLUMNS = {
    "FA": ["fa", "fly ash", "flyash"],
    "GGBS": ["ggbs", "slag", "ground granulated blast furnace slag"],
    "MK": ["mk", "metakaolin"],
    "NCA": ["nca", "natural coarse aggregate"],
    "NWA": ["nwa", "natural water absorption"],
    "RCA": ["rca", "recycled coarse aggregate"],
    "RWA": ["rwa", "recycled water absorption"],
    "NFA": ["nfa", "natural fine aggregate"],
    "NH": ["nh", "sodium hydroxide content", "naoh content"],
    "NHC": ["nhc", "sodium hydroxide concentration", "naoh concentration"],
    "NS": ["ns", "sodium silicate content"],
    "CT": ["ct", "curing temperature"],
    "HCD": ["hcd", "heat curing duration", "curing duration"],
    "SP": ["sp", "superplasticizer", "superplasticizer dosage"],
    "Age": ["curing age", "age"],
}
GEOPOLYMER_TARGET = [
    "cs", "compressive strength", "compressivestrength", "strength"
]


def extract_xy(df, feature_map, target_aliases, case_name, mean_impute=None):
    feature_cols = []
    resolved = {}
    for canonical, aliases in feature_map.items():
        c = find_column(df, [canonical] + aliases)
        feature_cols.append(c)
        resolved[canonical] = c

    target_col = find_column(df, target_aliases)

    work = df[feature_cols + [target_col]].copy()
    for c in work.columns:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    n_before = len(work)
    imputation_log = {}

    # Reproduce the published carbonation-data preprocessing:
    # replace missing T and RH entries with their respective column means.
    if mean_impute:
        for canonical in mean_impute:
            c = resolved[canonical]
            n_missing = int(work[c].isna().sum())
            mean_value = float(work[c].mean(skipna=True))
            work[c] = work[c].fillna(mean_value)
            imputation_log[canonical] = {
                "column": c,
                "missing_replaced": n_missing,
                "mean_used": mean_value
            }

    # Any missing value remaining in another predictor/target is not silently imputed.
    remaining = work.isna().sum()
    remaining = remaining[remaining > 0]
    if len(remaining):
        raise ValueError(
            f"Unexpected missing values remain in {case_name}:\n{remaining}\n"
            "Only the explicitly specified published mean-imputation scenario is allowed."
        )

    n_after = len(work)
    X = work[feature_cols].to_numpy(dtype=float)
    y = work[target_col].to_numpy(dtype=float)

    print("\n" + "=" * 80)
    print(case_name)
    print("=" * 80)
    print("Resolved predictors:")
    for k, v in resolved.items():
        print(f"  {k:28s} -> {v}")
    print(f"Target: {target_col}")
    print(f"Rows retained: {n_after}/{n_before}")
    if imputation_log:
        print("Mean imputation:")
        for k, v in imputation_log.items():
            print(f"  {k}: replaced {v['missing_replaced']} missing values "
                  f"with mean = {v['mean_used']:.6f}")
    print(f"X shape: {X.shape}; y shape: {y.shape}")

    return X, y, resolved, target_col, imputation_log


def decode_params(z):
    z = np.asarray(z, dtype=float)
    return {
        "C": 10.0 ** z[0],
        "gamma": 10.0 ** z[1],
        "epsilon": 10.0 ** z[2],
    }


class CachedCVObjective:
    """
    CV objective with caching and evaluation accounting.

    Hyperparameters are rounded in log-space only for the cache key;
    the actual unrounded candidate is used when first evaluated.
    """
    def __init__(self, X_train, y_train, cv):
        self.X = X_train
        self.y = y_train
        self.cv = cv
        self.cache = {}
        self.requests = 0
        self.computed = 0

    def __call__(self, z):
        self.requests += 1
        key = tuple(np.round(np.asarray(z, dtype=float), 10))
        if key in self.cache:
            return self.cache[key]

        p = decode_params(z)
        model = Pipeline([
            ("scale", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=p["C"],
                        gamma=p["gamma"], epsilon=p["epsilon"]))
        ])

        scores = cross_val_score(
            model, self.X, self.y,
            cv=self.cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=1
        )
        rmse = float(-np.mean(scores))
        self.cache[key] = rmse
        self.computed += 1
        return rmse


def run_optimizer(name, objective, seed):
    base = BaseOptParams(n_agents=N_AGENTS, iters=ITERS)

    if name == "GFO":
        p = GFOParams(n_agents=N_AGENTS, iters=ITERS, seed=seed)
        return GFO(objective, LB, UB, params=p, verbose=False).fit()
    if name == "PSO":
        return pso_opt(objective, LB, UB, base, seed=seed)
    if name == "DE":
        return de_opt(objective, LB, UB, base, seed=seed)
    if name == "GWO":
        return gwo_opt(objective, LB, UB, base, seed=seed)
    if name == "WOA":
        return woa_opt(objective, LB, UB, base, seed=seed)
    if name == "GA":
        return ga_opt(objective, LB, UB, base, seed=seed)
    if name == "RandomSearch":
        return random_search_opt(objective, LB, UB, base, seed=seed)
    raise ValueError(name)


def fit_and_test(best_z, X_train, y_train, X_test, y_test):
    p = decode_params(best_z)
    model = Pipeline([
        ("scale", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=p["C"],
                    gamma=p["gamma"], epsilon=p["epsilon"]))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return {
        "Test_RMSE": float(mean_squared_error(y_test, pred) ** 0.5),
        "Test_MAE": float(mean_absolute_error(y_test, pred)),
        "Test_R2": float(r2_score(y_test, pred)),
        "pred": pred,
        **p
    }


def holm_adjust(pvals):
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for k, idx in enumerate(order):
        value = (m-k) * pvals[idx]
        running = max(running, value)
        adj[idx] = min(1.0, running)
    return adj


def run_case(case_name, X, y):
    case_dir = OUTPUT_DIR / case_name.replace(" ", "_")
    case_dir.mkdir(parents=True, exist_ok=True)

    # One fixed 80/20 holdout for every optimizer/run.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=DATA_SPLIT_SEED
    )

    print(f"\nTrain/test: {len(y_train)}/{len(y_test)}")

    raw_path = case_dir / "raw_runs.csv"
    if raw_path.exists():
        raw = pd.read_csv(raw_path)
        print(f"Resume file found: {len(raw)} completed rows.")
    else:
        raw = pd.DataFrame()

    records = [] if raw.empty else raw.to_dict("records")
    completed = set()
    if not raw.empty:
        completed = set(zip(raw["Algorithm"], raw["Run"].astype(int)))

    histories = []

    for run in range(N_RUNS):
        seed = SEED0 + run

        # Fixed folds for all algorithms in this run.
        cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

        for alg in ALGORITHMS:
            if (alg, run) in completed:
                print(f"[SKIP] {case_name} | {alg} | run {run+1}/{N_RUNS}")
                continue

            print(f"[RUN ] {case_name} | {alg} | run {run+1}/{N_RUNS} | seed={seed}")
            objective = CachedCVObjective(X_train, y_train, cv)

            t0 = time.perf_counter()
            best_z, best_cv, info = run_optimizer(alg, objective, seed)
            runtime = time.perf_counter() - t0

            test = fit_and_test(best_z, X_train, y_train, X_test, y_test)

            rec = {
                "Case": case_name,
                "Algorithm": alg,
                "Run": run,
                "Seed": seed,
                "Best_CV_RMSE": float(best_cv),
                "Test_RMSE": test["Test_RMSE"],
                "Test_MAE": test["Test_MAE"],
                "Test_R2": test["Test_R2"],
                "C": test["C"],
                "gamma": test["gamma"],
                "epsilon": test["epsilon"],
                "Runtime_s": runtime,
                "Fitness_requests": objective.requests,
                "Computed_CV_evaluations": objective.computed,
            }
            records.append(rec)

            hist = np.asarray(info["history_best"], dtype=float)
            for it, value in enumerate(hist):
                histories.append({
                    "Case": case_name, "Algorithm": alg, "Run": run,
                    "Iteration": it, "Best_CV_RMSE": value
                })

            # Checkpoint after every run.
            pd.DataFrame(records).to_csv(raw_path, index=False)
            pd.DataFrame(histories).to_csv(case_dir/"convergence_new_runs.csv", index=False)

    raw = pd.DataFrame(records)
    raw.to_csv(raw_path, index=False)

    # ---------- Summary ----------
    summary = raw.groupby("Algorithm").agg(
        Mean_CV_RMSE=("Best_CV_RMSE","mean"),
        Std_CV_RMSE=("Best_CV_RMSE","std"),
        Mean_Test_RMSE=("Test_RMSE","mean"),
        Std_Test_RMSE=("Test_RMSE","std"),
        Mean_Test_MAE=("Test_MAE","mean"),
        Mean_Test_R2=("Test_R2","mean"),
        Mean_Runtime_s=("Runtime_s","mean"),
        Mean_Fitness_requests=("Fitness_requests","mean"),
        Mean_Computed_CV_evaluations=("Computed_CV_evaluations","mean"),
    ).reindex(ALGORITHMS)
    summary.to_csv(case_dir/"performance_summary.csv")

    # Representative hyperparameters:
    # run having the median CV-RMSE for each algorithm.
    reps = []
    for alg in ALGORITHMS:
        a = raw[raw.Algorithm == alg].sort_values("Best_CV_RMSE")
        medpos = len(a)//2
        r = a.iloc[medpos]
        reps.append({
            "Algorithm": alg,
            "CV_RMSE": r.Best_CV_RMSE,
            "C": r.C, "gamma": r.gamma, "epsilon": r.epsilon,
            "Test_RMSE": r.Test_RMSE, "Test_MAE": r.Test_MAE,
            "Test_R2": r.Test_R2
        })
    pd.DataFrame(reps).to_csv(case_dir/"representative_hyperparameters.csv", index=False)

    # ---------- Friedman across matched runs ----------
    pivot = raw.pivot(index="Run", columns="Algorithm", values="Best_CV_RMSE").dropna()
    fried_stat, fried_p = friedmanchisquare(*[pivot[a].values for a in ALGORITHMS])

    rank_matrix = np.vstack([
        rankdata(pivot.loc[r, ALGORITHMS].values, method="average")
        for r in pivot.index
    ])
    avg_ranks = rank_matrix.mean(axis=0)
    rank_df = pd.DataFrame({"Algorithm": ALGORITHMS, "Average_Rank": avg_ranks})
    rank_df = rank_df.sort_values("Average_Rank")
    rank_df.to_csv(case_dir/"friedman_ranks.csv", index=False)

    with open(case_dir/"friedman_test.json","w") as f:
        json.dump({
            "n_complete_runs": int(len(pivot)),
            "statistic": float(fried_stat),
            "p_value": float(fried_p)
        }, f, indent=2)

    # ---------- Paired Wilcoxon signed-rank: GFO vs each comparator ----------
    rows = []
    raw_p = []
    for comp in ALGORITHMS[1:]:
        x = pivot["GFO"].values
        yv = pivot[comp].values
        try:
            W, p = wilcoxon(x, yv, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            W, p = 0.0, 1.0

        d = x - yv  # lower is better
        wins = int(np.sum(d < -1e-12))
        ties = int(np.sum(np.abs(d) <= 1e-12))
        losses = int(np.sum(d > 1e-12))
        rows.append([comp, W, p, wins, ties, losses])
        raw_p.append(p)

    adj = holm_adjust(raw_p)
    wil = pd.DataFrame(rows, columns=[
        "Compared_Algorithm","W","Raw_p","Wins","Ties","Losses"
    ])
    wil["Holm_p"] = adj
    wil["Significant_0.05"] = wil["Holm_p"] < 0.05
    wil.to_csv(case_dir/"paired_wilcoxon_holm_vs_GFO.csv", index=False)

    print("\nSummary:")
    print(summary[["Mean_CV_RMSE","Std_CV_RMSE","Mean_Test_RMSE","Mean_Test_R2"]])
    print("\nFriedman ranks:")
    print(rank_df)
    print(f"Friedman statistic={fried_stat:.6g}, p={fried_p:.6g}")
    print("\nPaired Wilcoxon + Holm:")
    print(wil)

    return raw


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Carbonation
    cdf = read_table(CARBONATION_FILE, CARBONATION_SHEET)
    Xc, yc, cresolved, ctarget, cimpute = extract_xy(
        cdf, CARBONATION_COLUMNS, CARBONATION_TARGET,
        "Carbonation Depth", mean_impute=["Temperature", "RelativeHumidity"]
    )
    if len(yc) != 819:
        print(f"WARNING: carbonation usable sample count is {len(yc)}, not 819.")

    # Recycled geopolymer concrete
    gdf = read_table(GEOPOLYMER_FILE, GEOPOLYMER_SHEET)
    Xg, yg, gresolved, gtarget, gimpute = extract_xy(
        gdf, GEOPOLYMER_COLUMNS, GEOPOLYMER_TARGET,
        "Recycled Geopolymer Concrete"
    )
    # Manuscript currently describes 373 usable samples.
    if len(yg) != 373:
        print(f"WARNING: geopolymer usable sample count is {len(yg)}, not 373.")

    # Save resolved schema before expensive optimization.
    with open(OUTPUT_DIR/"resolved_columns.json","w") as f:
        json.dump({
            "Carbonation": {"features": cresolved, "target": ctarget, "imputation": cimpute},
            "Recycled_Geopolymer_Concrete": {"features": gresolved, "target": gtarget, "imputation": gimpute}
        }, f, indent=2)

    run_case("Carbonation_Depth", Xc, yc)
    run_case("Recycled_Geopolymer_Concrete", Xg, yg)

    print("\n" + "="*80)
    print("ENGINEERING EXPERIMENTS COMPLETE")
    print("="*80)
    print(f"Results saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
