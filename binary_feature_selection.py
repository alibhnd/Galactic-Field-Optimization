#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproducibility script for the FINAL field-preserving BGFO feature-selection study.

Purpose
-------
This file runs ONLY the revised binary feature-selection experiments.
The continuous benchmarks and ablation study remain frozen and are not rerun.

Main improvements over the previous all-in-one script
------------------------------------------------------
1. Explicit progress printing for every dataset / optimizer / repetition.
2. A CSV checkpoint is written after EVERY completed repetition.
3. Safe resume: rerunning the script skips completed repetitions.
4. OpenML certificate handling uses certifi; SSL verification is NOT disabled.
5. Sparse OpenML ARFF datasets are handled correctly with as_frame='auto'.
6. OpenML datasets are cached locally.
7. Fitness values are memoized within each optimizer run to avoid repeated
   k-NN cross-validation for identical feature masks.
8. Final score, selected-feature, Wilcoxon, Friedman-rank, dataset-summary,
   and raw-run CSV files are saved automatically.

Recommended PyCharm environment
-------------------------------
Python 3.11, 3.12, or 3.13

Install:
    pip install numpy pandas scipy scikit-learn certifi

Then run this file directly in PyCharm.
"""

from __future__ import annotations

import os
import ssl
import time
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

import certifi
import numpy as np
import pandas as pd

from scipy.stats import friedmanchisquare, rankdata, wilcoxon
from scipy import sparse

from sklearn.datasets import (
    fetch_openml,
    load_breast_cancer,
    load_wine,
    load_iris,
    load_digits,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")



from gfo.binary import (
    CachedFeatureSelectionFitness,
    FieldPreservingBGFOParams,
    FieldPreservingBinaryGFO,
    BinaryPSO,
    BinaryGA,
    BinaryRandomSearch,
)

# ======================================================================
# USER SETTINGS
# ======================================================================

OUTPUT_DIR = Path("results/binary_feature_selection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OPENML_CACHE_DIR = OUTPUT_DIR / "openml_cache"
OPENML_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Experimental settings retained from the previous study.
# IMPORTANT: this script uses a NEW output directory/checkpoint because BGFO changed.
ALGOS = ["BGFO", "BPSO", "BGA", "BRS"]
R = 10
SEED0 = 42
N_AGENTS = 20
ITERS = 150
ALPHA = 0.99
BETA = 0.01
N_SPLITS = 5
N_NEIGHBORS = 5

# If True, the program stops before optimization when one or more requested
# OpenML datasets cannot be loaded. For a final paper-quality run, True is
# recommended so you do not unknowingly analyze an incomplete dataset set.
REQUIRE_ALL_DATASETS = True

# Number of fetch attempts for an OpenML dataset.
OPENML_RETRIES = 3

# If you want a quick test before the full run, change these temporarily:
# R = 2
# N_AGENTS = 8
# ITERS = 20


# ======================================================================
# CERTIFICATE SETUP
# ======================================================================

def configure_ssl_certificates() -> None:
    """
    Point Python/OpenML HTTPS requests to certifi's CA bundle.
    Verification remains enabled.
    """
    ca_file = certifi.where()
    os.environ["SSL_CERT_FILE"] = ca_file
    os.environ["REQUESTS_CA_BUNDLE"] = ca_file

    # Validate that the CA file can be loaded.
    ssl.create_default_context(cafile=ca_file)

    print("=" * 78)
    print("SSL certificate configuration")
    print("=" * 78)
    print(f"certifi CA bundle: {ca_file}")
    print("SSL verification: ENABLED")
    print()


# ======================================================================
# DATASET HELPERS
# ======================================================================

def _to_numeric_dataset(X, y):
    """
    Convert OpenML output to a dense numeric matrix safely.

    Handles:
    - pandas DataFrame / Series
    - NumPy arrays
    - SciPy sparse matrices (e.g., Australian dataset)
    """

    # ----- Features -----
    if sparse.issparse(X):
        X_num = X.toarray().astype(float)

        # Replace infinities with NaN before imputation.
        X_num[~np.isfinite(X_num)] = np.nan

        X_num = SimpleImputer(
            strategy="most_frequent"
        ).fit_transform(X_num)

    else:
        X_df = pd.DataFrame(X).copy()

        for col in X_df.columns:
            if not pd.api.types.is_numeric_dtype(X_df[col]):
                X_df[col] = pd.Categorical(
                    X_df[col]
                ).codes

        X_df = X_df.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        X_num = SimpleImputer(
            strategy="most_frequent"
        ).fit_transform(X_df)

    # ----- Target -----
    if sparse.issparse(y):
        y_arr = np.asarray(
            y.toarray()
        ).ravel()
        y_series = pd.Series(y_arr)
    else:
        y_series = pd.Series(
            np.asarray(y).ravel()
        )

    if not pd.api.types.is_numeric_dtype(
        y_series
    ):
        y_num = LabelEncoder().fit_transform(
            y_series.astype(str)
        )
    else:
        y_num = y_series.to_numpy()

    return (
        np.asarray(X_num, dtype=float),
        np.asarray(y_num),
    )


def fetch_openml_with_retry(
    name: str,
    version: int = 1,
    retries: int = OPENML_RETRIES,
):
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            print(
                f"  Fetching OpenML dataset '{name}' "
                f"(attempt {attempt}/{retries}) ..."
            )

            data = fetch_openml(
                name=name,
                version=version,
                as_frame="auto",
                data_home=str(OPENML_CACHE_DIR),
                parser="auto",
            )

            X, y = _to_numeric_dataset(data.data, data.target)

            print(
                f"    Loaded: {X.shape[0]} instances, "
                f"{X.shape[1]} features, "
                f"{len(np.unique(y))} classes"
            )
            return X, y, None

        except Exception as exc:
            last_error = exc
            print(f"    Failed: {type(exc).__name__}: {exc}")

            if attempt < retries:
                wait = 2 * attempt
                print(f"    Retrying in {wait} s ...")
                time.sleep(wait)

    return None, None, last_error


def load_feature_selection_datasets():
    """
    Load the four built-in sklearn datasets plus 14 OpenML datasets
    used in the previous feature-selection experiment.
    """
    datasets = {}

    print("=" * 78)
    print("Loading built-in datasets")
    print("=" * 78)

    data = load_breast_cancer()
    datasets["BreastCancer"] = (
        np.asarray(data.data, dtype=float),
        np.asarray(data.target),
    )

    data = load_wine()
    datasets["Wine"] = (
        np.asarray(data.data, dtype=float),
        np.asarray(data.target),
    )

    data = load_iris()
    datasets["Iris"] = (
        np.asarray(data.data, dtype=float),
        np.asarray(data.target),
    )

    data = load_digits()
    datasets["Digits"] = (
        np.asarray(data.data, dtype=float),
        np.asarray(data.target),
    )

    for name, (X, y) in datasets.items():
        print(
            f"  {name:18s}: "
            f"{X.shape[0]:5d} instances, "
            f"{X.shape[1]:4d} features, "
            f"{len(np.unique(y)):2d} classes"
        )

    openml_specs = [
        ("ionosphere", 1, "Ionosphere"),
        ("sonar", 1, "Sonar"),
        ("vehicle", 1, "Vehicle"),
        ("glass", 1, "Glass"),
        ("dermatology", 1, "Dermatology"),
        ("wine-quality-red", 1, "WineQualityRed"),
        ("wine-quality-white", 1, "WineQualityWhite"),
        ("australian", 1, "Australian"),
        ("heart-statlog", 1, "HeartStatlog"),
        ("blood-transfusion-service-center", 1, "BloodTransfusion"),
        ("cylinder-bands", 1, "CylinderBands"),
        ("diabetes", 1, "Diabetes"),
        ("ecoli", 1, "Ecoli"),
        ("balance-scale", 1, "BalanceScale"),
    ]

    failures = []

    print()
    print("=" * 78)
    print("Loading OpenML datasets")
    print("=" * 78)

    for openml_name, version, alias in openml_specs:
        X, y, error = fetch_openml_with_retry(
            openml_name,
            version=version,
        )

        if X is not None:
            datasets[alias] = (X, y)
        else:
            failures.append(
                {
                    "alias": alias,
                    "openml_name": openml_name,
                    "version": version,
                    "error": repr(error),
                }
            )

    failure_path = OUTPUT_DIR / "dataset_load_failures.csv"
    pd.DataFrame(failures).to_csv(failure_path, index=False)

    print()
    print("=" * 78)
    print("Dataset loading summary")
    print("=" * 78)
    print(f"Successfully loaded: {len(datasets)}")
    print(f"Failed:              {len(failures)}")

    if failures:
        print("\nFailed datasets:")
        for x in failures:
            print(f"  - {x['alias']}: {x['error']}")

        print(f"\nFailure log: {failure_path.resolve()}")

        if REQUIRE_ALL_DATASETS:
            raise RuntimeError(
                "\nOne or more requested datasets could not be loaded.\n"
                "The optimization was NOT started because "
                "REQUIRE_ALL_DATASETS=True.\n"
                "This prevents an incomplete paper-quality experiment.\n"
                "Fix the listed dataset-loading issue, then rerun."
            )

    return datasets


# ======================================================================
# ONE RUN
# ======================================================================

def run_binary_optimizer_once(
    algo_name: str,
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
):
    fitness = CachedFeatureSelectionFitness(
        X=X,
        y=y,
        alpha=ALPHA,
        beta=BETA,
        n_splits=N_SPLITS,
        seed=seed,
        n_neighbors=N_NEIGHBORS,
    )

    d = X.shape[1]

    if algo_name == "BGFO":
        params = FieldPreservingBGFOParams(
            n_agents=N_AGENTS,
            iters=ITERS,
            seed=seed,
        )
        opt = FieldPreservingBinaryGFO(
            fitness=fitness,
            n_features=d,
            params=params,
        )

    elif algo_name == "BPSO":
        opt = BinaryPSO(
            fitness=fitness,
            n_features=d,
            n_agents=N_AGENTS,
            iters=ITERS,
            seed=seed,
        )

    elif algo_name == "BGA":
        opt = BinaryGA(
            fitness=fitness,
            n_features=d,
            n_agents=N_AGENTS,
            iters=ITERS,
            seed=seed,
        )

    elif algo_name == "BRS":
        opt = BinaryRandomSearch(
            fitness=fitness,
            n_features=d,
            n_agents=N_AGENTS,
            iters=ITERS,
            seed=seed,
        )

    else:
        raise ValueError(
            f"Unknown algorithm: {algo_name}"
        )

    result = opt.fit()

    result["fitness_requests"] = (
        fitness.n_requested
    )
    result["fitness_computed"] = (
        fitness.n_computed
    )
    result["cache_hit_rate"] = (
        fitness.cache_hit_rate
    )

    return result


# ======================================================================
# CHECKPOINT / RESUME
# ======================================================================

RAW_RUNS_CSV = OUTPUT_DIR / "field_preserving_binary_fs_raw_runs.csv"


def load_checkpoint() -> pd.DataFrame:
    if RAW_RUNS_CSV.exists():
        df = pd.read_csv(RAW_RUNS_CSV)

        print(
            f"Checkpoint found: "
            f"{len(df)} completed runs"
        )
        print(
            f"  {RAW_RUNS_CSV.resolve()}"
        )

        return df

    return pd.DataFrame(
        columns=[
            "Dataset",
            "Algorithm",
            "Run",
            "Seed",
            "BestScore",
            "SelectedFeatures",
            "RuntimeSeconds",
            "FitnessRequests",
            "FitnessComputed",
            "CacheHitRate",
        ]
    )


def save_checkpoint(
    df: pd.DataFrame,
) -> None:
    # Atomic-style write through a temporary file.
    temp = RAW_RUNS_CSV.with_suffix(".tmp.csv")
    df.to_csv(temp, index=False)
    temp.replace(RAW_RUNS_CSV)


def already_completed(
    df: pd.DataFrame,
    dataset: str,
    algo: str,
    run: int,
) -> bool:
    if len(df) == 0:
        return False

    m = (
        (df["Dataset"] == dataset)
        & (df["Algorithm"] == algo)
        & (df["Run"].astype(int) == int(run))
    )

    return bool(m.any())


# ======================================================================
# MAIN REPEATED EXPERIMENT WITH PROGRESS
# ======================================================================

def run_experiments(
    datasets: dict,
) -> pd.DataFrame:
    checkpoint = load_checkpoint()

    total_runs = (
        len(datasets)
        * len(ALGOS)
        * R
    )

    already = 0
    for dname in datasets:
        for algo in ALGOS:
            for r in range(R):
                if already_completed(
                    checkpoint,
                    dname,
                    algo,
                    r + 1,
                ):
                    already += 1

    print()
    print("=" * 78)
    print("Binary feature-selection experiment")
    print("=" * 78)
    print(f"Datasets:             {len(datasets)}")
    print(f"Algorithms:           {len(ALGOS)}")
    print(f"Repetitions:          {R}")
    print(f"Agents:               {N_AGENTS}")
    print(f"Iterations:           {ITERS}")
    print(f"CV folds:             {N_SPLITS}")
    print(f"Total planned runs:   {total_runs}")
    print(f"Already completed:    {already}")
    print(f"Remaining:            {total_runs - already}")
    print()

    experiment_start = time.perf_counter()
    completed_this_session = 0

    dnames = list(datasets.keys())

    for d_idx, dname in enumerate(
        dnames,
        start=1,
    ):
        X, y = datasets[dname]

        print()
        print("#" * 78)
        print(
            f"[DATASET {d_idx}/{len(dnames)}] "
            f"{dname}"
        )
        print(
            f"Instances={X.shape[0]}, "
            f"Features={X.shape[1]}, "
            f"Classes={len(np.unique(y))}"
        )
        print("#" * 78)

        for a_idx, algo in enumerate(
            ALGOS,
            start=1,
        ):
            print()
            print(
                f"  [ALGORITHM {a_idx}/{len(ALGOS)}] "
                f"{algo}"
            )

            for r in range(R):
                run_number = r + 1
                seed = SEED0 + r

                if already_completed(
                    checkpoint,
                    dname,
                    algo,
                    run_number,
                ):
                    row = checkpoint[
                        (checkpoint["Dataset"] == dname)
                        & (checkpoint["Algorithm"] == algo)
                        & (
                            checkpoint["Run"].astype(int)
                            == run_number
                        )
                    ].iloc[0]

                    print(
                        f"    Run {run_number:02d}/{R}: "
                        f"SKIPPED (checkpoint) | "
                        f"score={row['BestScore']:.6f}"
                    )
                    continue

                run_start = time.perf_counter()

                print(
                    f"    Run {run_number:02d}/{R}: "
                    f"starting ...",
                    flush=True,
                )

                result = run_binary_optimizer_once(
                    algo_name=algo,
                    X=X,
                    y=y,
                    seed=seed,
                )

                elapsed = (
                    time.perf_counter()
                    - run_start
                )

                new_row = {
                    "Dataset": dname,
                    "Algorithm": algo,
                    "Run": run_number,
                    "Seed": seed,
                    "BestScore": result[
                        "best_score"
                    ],
                    "SelectedFeatures": result[
                        "selected_features"
                    ],
                    "RuntimeSeconds": elapsed,
                    "FitnessRequests": result[
                        "fitness_requests"
                    ],
                    "FitnessComputed": result[
                        "fitness_computed"
                    ],
                    "CacheHitRate": result[
                        "cache_hit_rate"
                    ],
                }

                checkpoint = pd.concat(
                    [
                        checkpoint,
                        pd.DataFrame([new_row]),
                    ],
                    ignore_index=True,
                )

                save_checkpoint(checkpoint)

                completed_this_session += 1

                done_total = (
                    already
                    + completed_this_session
                )

                rate = (
                    (
                        time.perf_counter()
                        - experiment_start
                    )
                    / max(
                        completed_this_session,
                        1,
                    )
                )

                remaining = (
                    total_runs - done_total
                )

                eta_seconds = rate * remaining

                print(
                    f"    Run {run_number:02d}/{R}: DONE | "
                    f"score={result['best_score']:.6f} | "
                    f"features={result['selected_features']} | "
                    f"time={elapsed:.1f}s | "
                    f"cache hit={100*result['cache_hit_rate']:.1f}%"
                )

                print(
                    f"      Overall progress: "
                    f"{done_total}/{total_runs} "
                    f"({100*done_total/total_runs:.1f}%) | "
                    f"approx ETA={eta_seconds/3600:.2f} h"
                )

    return checkpoint


# ======================================================================
# SUMMARY / STATISTICS
# ======================================================================

def build_score_table(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for dname in sorted(
        raw["Dataset"].unique()
    ):
        row = {"Dataset": dname}

        for algo in ALGOS:
            x = raw.loc[
                (
                    raw["Dataset"] == dname
                )
                & (
                    raw["Algorithm"] == algo
                ),
                "BestScore",
            ].to_numpy(dtype=float)

            if len(x) == 0:
                row[algo] = ""
            else:
                row[algo] = (
                    f"{np.mean(x):.4f} ± "
                    f"{np.std(x, ddof=1) if len(x)>1 else 0.0:.4f}"
                )

        rows.append(row)

    return pd.DataFrame(rows)


def build_feature_table(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for dname in sorted(
        raw["Dataset"].unique()
    ):
        row = {"Dataset": dname}

        for algo in ALGOS:
            x = raw.loc[
                (
                    raw["Dataset"] == dname
                )
                & (
                    raw["Algorithm"] == algo
                ),
                "SelectedFeatures",
            ].to_numpy(dtype=float)

            if len(x) == 0:
                row[algo] = ""
            else:
                row[algo] = (
                    f"{np.mean(x):.2f} ± "
                    f"{np.std(x, ddof=1) if len(x)>1 else 0.0:.2f}"
                )

        rows.append(row)

    return pd.DataFrame(rows)


def _safe_paired_wilcoxon(x, y):
    """Paired two-sided Wilcoxon signed-rank test with exact-tie handling."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diff = x - y

    if len(diff) == 0:
        return np.nan, np.nan
    if np.all(np.isclose(diff, 0.0, rtol=1e-12, atol=1e-15)):
        return 0.0, 1.0

    try:
        stat, p = wilcoxon(
            x,
            y,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method="auto",
        )
        return float(stat), float(p)
    except ValueError:
        return np.nan, 1.0


def _holm_adjust(p_values):
    """Holm step-down multiplicity adjustment."""
    p = np.asarray(p_values, dtype=float)
    if len(p) == 0:
        return np.asarray([], dtype=float)

    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    running = 0.0

    for rank, idx in enumerate(order):
        value = (len(p) - rank) * p[idx]
        running = max(running, value)
        adjusted[idx] = min(running, 1.0)

    return adjusted


def paired_wilcoxon_by_dataset_vs_bgfo(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Paired BGFO-vs-comparator tests within each dataset using matched run numbers.

    Holm correction is applied within each dataset across the three comparator
    tests. This table is supplementary to the global manuscript comparison.
    """
    rows = []

    for dname in sorted(raw["Dataset"].unique()):
        d = raw[raw["Dataset"] == dname]

        for algo in ALGOS:
            if algo == "BGFO":
                continue

            ref = (
                d[d["Algorithm"] == "BGFO"][["Run", "BestScore"]]
                .rename(columns={"BestScore": "BGFO"})
            )
            comp = (
                d[d["Algorithm"] == algo][["Run", "BestScore"]]
                .rename(columns={"BestScore": algo})
            )
            paired = ref.merge(comp, on="Run", how="inner").sort_values("Run")

            if len(paired) != R:
                continue

            x = paired["BGFO"].to_numpy(dtype=float)
            y = paired[algo].to_numpy(dtype=float)
            stat, p = _safe_paired_wilcoxon(x, y)

            diff = x - y  # lower objective is better
            rows.append({
                "Dataset": dname,
                "Reference": "BGFO",
                "Compared": algo,
                "W": stat,
                "Raw_p": p,
                "Wins": int(np.sum(diff < -1e-12)),
                "Ties": int(np.sum(np.abs(diff) <= 1e-12)),
                "Losses": int(np.sum(diff > 1e-12)),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["Holm_p"] = np.nan
    for dname in out["Dataset"].unique():
        idx = out.index[out["Dataset"] == dname]
        out.loc[idx, "Holm_p"] = _holm_adjust(
            out.loc[idx, "Raw_p"].to_numpy(dtype=float)
        )
    out["Significant_0.05"] = out["Holm_p"] < 0.05
    return out


def global_paired_wilcoxon_vs_bgfo(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Global paired Wilcoxon tests used for the manuscript comparison.

    Each dataset contributes one mean wrapper-objective value per algorithm.
    Pairing is therefore across the same 18 datasets, followed by Holm
    correction across the three BGFO-vs-comparator comparisons.
    """
    means = (
        raw.groupby(["Dataset", "Algorithm"], as_index=False)["BestScore"]
        .mean()
        .pivot(index="Dataset", columns="Algorithm", values="BestScore")
    )

    means = means.dropna(subset=ALGOS)
    rows = []
    raw_p = []

    for algo in ALGOS:
        if algo == "BGFO":
            continue

        x = means["BGFO"].to_numpy(dtype=float)
        y = means[algo].to_numpy(dtype=float)
        stat, p = _safe_paired_wilcoxon(x, y)
        diff = x - y  # lower objective is better

        rows.append({
            "Reference": "BGFO",
            "Compared": algo,
            "W": stat,
            "Raw_p": p,
            "Wins": int(np.sum(diff < -1e-12)),
            "Ties": int(np.sum(np.abs(diff) <= 1e-12)),
            "Losses": int(np.sum(diff > 1e-12)),
            "N_datasets": int(len(means)),
        })
        raw_p.append(p)

    out = pd.DataFrame(rows)
    out["Holm_p"] = _holm_adjust(raw_p)
    out["Significant_0.05"] = out["Holm_p"] < 0.05
    return out

def friedman_average_ranks(
    raw: pd.DataFrame,
):
    valid_datasets = []

    for dname in sorted(
        raw["Dataset"].unique()
    ):
        ok = True

        for algo in ALGOS:
            n = len(
                raw[
                    (
                        raw["Dataset"] == dname
                    )
                    & (
                        raw["Algorithm"] == algo
                    )
                ]
            )

            if n != R:
                ok = False
                break

        if ok:
            valid_datasets.append(dname)

    if len(valid_datasets) < 2:
        return {
            "friedman_stat": np.nan,
            "friedman_p": np.nan,
            "rank_table": pd.DataFrame(),
            "n_complete_datasets": len(
                valid_datasets
            ),
        }

    M = np.zeros(
        (
            len(valid_datasets),
            len(ALGOS),
        ),
        dtype=float,
    )

    for i, dname in enumerate(
        valid_datasets
    ):
        for j, algo in enumerate(ALGOS):
            M[i, j] = raw.loc[
                (
                    raw["Dataset"] == dname
                )
                & (
                    raw["Algorithm"] == algo
                ),
                "BestScore",
            ].mean()

    ranks = np.vstack(
        [
            rankdata(
                M[i],
                method="average",
            )
            for i in range(
                M.shape[0]
            )
        ]
    )

    avg_ranks = np.mean(
        ranks,
        axis=0,
    )

    stat, p = friedmanchisquare(
        *[
            M[:, j]
            for j in range(
                M.shape[1]
            )
        ]
    )

    rank_df = pd.DataFrame(
        {
            "Algorithm": ALGOS,
            "AverageRank": avg_ranks,
        }
    ).sort_values(
        "AverageRank"
    ).reset_index(drop=True)

    return {
        "friedman_stat": float(stat),
        "friedman_p": float(p),
        "rank_table": rank_df,
        "n_complete_datasets": len(
            valid_datasets
        ),
    }


def save_final_outputs(
    raw: pd.DataFrame,
    datasets: dict,
):
    dataset_rows = []

    for name, (X, y) in datasets.items():
        dataset_rows.append(
            {
                "Dataset": name,
                "Instances": int(X.shape[0]),
                "Features": int(X.shape[1]),
                "Classes": int(
                    len(np.unique(y))
                ),
            }
        )

    dataset_summary = (
        pd.DataFrame(dataset_rows)
        .sort_values("Dataset")
        .reset_index(drop=True)
    )

    score_table = build_score_table(raw)
    feature_table = build_feature_table(raw)
    wilcox_table = paired_wilcoxon_by_dataset_vs_bgfo(raw)
    global_wilcox_table = global_paired_wilcoxon_vs_bgfo(raw)
    friedman_out = friedman_average_ranks(raw)

    dataset_summary.to_csv(
        OUTPUT_DIR
        / "binary_fs_dataset_summary.csv",
        index=False,
    )

    score_table.to_csv(
        OUTPUT_DIR
        / "binary_fs_score_summary.csv",
        index=False,
    )

    feature_table.to_csv(
        OUTPUT_DIR
        / "binary_fs_feature_summary.csv",
        index=False,
    )

    wilcox_table.to_csv(
        OUTPUT_DIR / "binary_fs_paired_wilcoxon_by_dataset_vs_BGFO.csv",
        index=False,
    )

    global_wilcox_table.to_csv(
        OUTPUT_DIR / "binary_fs_global_paired_wilcoxon_holm_vs_BGFO.csv",
        index=False,
    )

    friedman_out[
        "rank_table"
    ].to_csv(
        OUTPUT_DIR
        / "binary_fs_friedman_ranks.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR
        / "binary_fs_friedman_test.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "friedman_stat":
                    friedman_out[
                        "friedman_stat"
                    ],
                "friedman_p":
                    friedman_out[
                        "friedman_p"
                    ],
                "n_complete_datasets":
                    friedman_out[
                        "n_complete_datasets"
                    ],
            },
            f,
            indent=2,
        )

    print()
    print("=" * 78)
    print("FINAL SUMMARY")
    print("=" * 78)

    print("\nDataset summary:")
    print(dataset_summary.to_string(index=False))

    print("\nScore summary:")
    print(score_table.to_string(index=False))

    print("\nSelected-feature summary:")
    print(feature_table.to_string(index=False))

    print("\nFriedman ranks:")
    print(
        friedman_out[
            "rank_table"
        ].to_string(index=False)
    )

    print("\nGlobal paired Wilcoxon tests (dataset-level means, Holm-adjusted):")
    print(global_wilcox_table.to_string(index=False))

    print(
        "\nFriedman statistic:",
        friedman_out[
            "friedman_stat"
        ],
    )
    print(
        "Friedman p-value:",
        friedman_out[
            "friedman_p"
        ],
    )

    print()
    print(
        f"All outputs saved in:\n"
        f"{OUTPUT_DIR.resolve()}"
    )


# ======================================================================
# MAIN
# ======================================================================

def main():
    print()
    print("=" * 78)
    print("GFO MACHINE-LEARNING FEATURE-SELECTION STUDY")
    print("=" * 78)
    print(
        "This script can be stopped and restarted. "
        "Completed repetitions are preserved in the checkpoint."
    )
    print()

    configure_ssl_certificates()

    datasets = load_feature_selection_datasets()

    # Save dataset summary immediately after successful loading.
    dataset_rows = []
    for name, (X, y) in datasets.items():
        dataset_rows.append(
            {
                "Dataset": name,
                "Instances": int(X.shape[0]),
                "Features": int(X.shape[1]),
                "Classes": int(
                    len(np.unique(y))
                ),
            }
        )

    pd.DataFrame(
        dataset_rows
    ).sort_values(
        "Dataset"
    ).to_csv(
        OUTPUT_DIR
        / "binary_fs_dataset_summary.csv",
        index=False,
    )

    raw = run_experiments(datasets)

    save_final_outputs(
        raw=raw,
        datasets=datasets,
    )

    print()
    print("=" * 78)
    print("RUN COMPLETE")
    print("=" * 78)
    print(
        "If PyCharm also shows "
        "'Process finished with exit code 0', "
        "the script completed successfully."
    )


if __name__ == "__main__":
    main()