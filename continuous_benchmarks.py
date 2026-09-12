#!/usr/bin/env python3
"""Reproduce the final continuous-optimization study reported in the GFO manuscript."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

from gfo.continuous import *

OUTPUT_DIR = Path("results/continuous")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DIR = OUTPUT_DIR / "histories"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)
RAW_RUNS_CSV = OUTPUT_DIR / "continuous_raw_runs.csv"

SEED0 = 42
R = 20
N_AGENTS = 30
ITERS = 300
ALPHA = 0.05
ALGOS = ["GFO", "PSO", "DE", "GWO", "WOA", "GA", "RandomSearch"]
EXCLUDED_FUNCTIONS = {"PermDBeta", "Alpine2"}

# ======================================================================
# UNIFIED OPTIMIZER RUNNER
# ======================================================================

def run_optimizer_once(
    algo_name: str,
    func: Callable[[np.ndarray], float],
    lb: float,
    ub: float,
    D: int,
    seed: int,
    iters: int,
    n_agents: int,
):
    lbv = np.full(D, lb, dtype=float)
    ubv = np.full(D, ub, dtype=float)

    counted = CountingObjective(func)

    if algo_name == "GFO":
        P = GFOParams(
            n_agents=n_agents,
            iters=iters,
            seed=seed,
        )
        opt = GFO(
            counted,
            lbv,
            ubv,
            params=P,
            verbose=False,
        )
        _, best_f, info = opt.fit()

    else:
        P = BaseOptParams(
            n_agents=n_agents,
            iters=iters,
        )

        if algo_name == "PSO":
            _, best_f, info = pso_opt(counted, lbv, ubv, P, seed=seed)
        elif algo_name == "DE":
            _, best_f, info = de_opt(counted, lbv, ubv, P, seed=seed)
        elif algo_name == "GWO":
            _, best_f, info = gwo_opt(counted, lbv, ubv, P, seed=seed)
        elif algo_name == "WOA":
            _, best_f, info = woa_opt(counted, lbv, ubv, P, seed=seed)
        elif algo_name == "GA":
            _, best_f, info = ga_opt(counted, lbv, ubv, P, seed=seed)
        elif algo_name == "RandomSearch":
            _, best_f, info = random_search_opt(counted, lbv, ubv, P, seed=seed)
        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")

    return (
        float(best_f),
        np.asarray(info["history_best"], dtype=float),
        int(counted.count),
    )


# ======================================================================
# CHECKPOINT / RESUME
# ======================================================================

CHECKPOINT_COLUMNS = [
    "Function",
    "Category",
    "Dimension",
    "Algorithm",
    "Run",
    "Seed",
    "BestFitness",
    "RuntimeSeconds",
    "FunctionEvaluations",
    "HistoryFile",
]


def load_checkpoint() -> pd.DataFrame:
    if RAW_RUNS_CSV.exists():
        df = pd.read_csv(RAW_RUNS_CSV)
        print(f"Checkpoint found: {len(df)} completed runs")
        print(f"  {RAW_RUNS_CSV.resolve()}")
        return df

    return pd.DataFrame(columns=CHECKPOINT_COLUMNS)


def save_checkpoint(df: pd.DataFrame) -> None:
    tmp = RAW_RUNS_CSV.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(RAW_RUNS_CSV)


def is_completed(
    df: pd.DataFrame,
    function_name: str,
    algo: str,
    run_number: int,
) -> bool:
    if df.empty:
        return False

    mask = (
        (df["Function"] == function_name)
        & (df["Algorithm"] == algo)
        & (df["Run"].astype(int) == int(run_number))
    )
    return bool(mask.any())


def history_path(function_name: str, algo: str, run_number: int) -> Path:
    safe_fn = re.sub(r"[^A-Za-z0-9_-]+", "_", function_name)
    safe_alg = re.sub(r"[^A-Za-z0-9_-]+", "_", algo)
    return HISTORY_DIR / f"{safe_fn}__{safe_alg}__run{run_number:02d}.npz"


# ======================================================================
# MAIN EXPERIMENT
# ======================================================================

def run_experiment() -> pd.DataFrame:
    checkpoint = load_checkpoint()

    total_runs = len(BENCHMARKS) * len(ALGOS) * R
    existing = 0

    for fname in BENCHMARKS:
        for algo in ALGOS:
            for run_number in range(1, R + 1):
                if is_completed(checkpoint, fname, algo, run_number):
                    existing += 1

    print()
    print("=" * 80)
    print("FINAL GFO CONTINUOUS-OPTIMIZATION COMPARISON")
    print("=" * 80)
    print(f"Functions:             {len(BENCHMARKS)}")
    print(f"Algorithms:            {len(ALGOS)}")
    print(f"Independent runs:      {R}")
    print(f"Population size:       {N_AGENTS}")
    print(f"Iterations:            {ITERS}")
    print(f"Planned runs:          {total_runs}")
    print(f"Already completed:     {existing}")
    print(f"Remaining:             {total_runs - existing}")
    print(f"Output directory:      {OUTPUT_DIR.resolve()}")
    print()

    session_start = time.perf_counter()
    completed_session = 0

    functions = list(BENCHMARKS.items())

    for f_idx, (fname, spec) in enumerate(functions, start=1):
        func, lb, ub, D, category = spec

        print()
        print("#" * 80)
        print(
            f"[FUNCTION {f_idx}/{len(functions)}] "
            f"{fname} | category={category} | D={D} | bounds=[{lb}, {ub}]"
        )
        print("#" * 80)

        for a_idx, algo in enumerate(ALGOS, start=1):
            print(f"\n  [ALGORITHM {a_idx}/{len(ALGOS)}] {algo}")

            for r in range(R):
                run_number = r + 1
                seed = SEED0 + r

                if is_completed(checkpoint, fname, algo, run_number):
                    row = checkpoint[
                        (checkpoint["Function"] == fname)
                        & (checkpoint["Algorithm"] == algo)
                        & (checkpoint["Run"].astype(int) == run_number)
                    ].iloc[0]

                    print(
                        f"    Run {run_number:02d}/{R}: SKIPPED | "
                        f"best={float(row['BestFitness']):.6g}"
                    )
                    continue

                print(
                    f"    Run {run_number:02d}/{R}: starting ...",
                    flush=True,
                )

                start = time.perf_counter()

                best_f, history, n_evals = run_optimizer_once(
                    algo_name=algo,
                    func=func,
                    lb=lb,
                    ub=ub,
                    D=D,
                    seed=seed,
                    iters=ITERS,
                    n_agents=N_AGENTS,
                )

                elapsed = time.perf_counter() - start

                hpath = history_path(fname, algo, run_number)
                np.savez_compressed(
                    hpath,
                    history_best=history,
                    function=fname,
                    algorithm=algo,
                    run=run_number,
                    seed=seed,
                )

                new_row = {
                    "Function": fname,
                    "Category": category,
                    "Dimension": D,
                    "Algorithm": algo,
                    "Run": run_number,
                    "Seed": seed,
                    "BestFitness": best_f,
                    "RuntimeSeconds": elapsed,
                    "FunctionEvaluations": n_evals,
                    "HistoryFile": str(hpath),
                }

                checkpoint = pd.concat(
                    [checkpoint, pd.DataFrame([new_row])],
                    ignore_index=True,
                )
                save_checkpoint(checkpoint)

                completed_session += 1
                done_total = existing + completed_session

                elapsed_session = time.perf_counter() - session_start
                avg_per_new_run = elapsed_session / max(completed_session, 1)
                remaining = total_runs - done_total
                eta = avg_per_new_run * remaining

                print(
                    f"    Run {run_number:02d}/{R}: DONE | "
                    f"best={best_f:.6g} | "
                    f"evals={n_evals} | "
                    f"time={fmt_seconds(elapsed)}"
                )
                print(
                    f"      Progress {done_total}/{total_runs} "
                    f"({100*done_total/total_runs:.1f}%) | "
                    f"approx ETA={fmt_seconds(eta)}"
                )

    return checkpoint


# ======================================================================
# FINAL TABLES AND STATISTICS
# ======================================================================

def complete_functions(raw: pd.DataFrame):
    out = []
    for fname in BENCHMARKS:
        ok = True
        for algo in ALGOS:
            n = len(
                raw[
                    (raw["Function"] == fname)
                    & (raw["Algorithm"] == algo)
                ]
            )
            if n != R:
                ok = False
                break
        if ok:
            out.append(fname)
    return out


def build_mean_sd_table(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fname, (_, _, _, D, category) in BENCHMARKS.items():
        row = {
            "Function": fname,
            "Category": category,
            "Dimension": D,
        }
        for algo in ALGOS:
            vals = raw.loc[
                (raw["Function"] == fname)
                & (raw["Algorithm"] == algo),
                "BestFitness",
            ].to_numpy(dtype=float)

            if len(vals):
                sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                row[algo] = f"{np.mean(vals):.6g} ± {sd:.6g}"
            else:
                row[algo] = ""
        rows.append(row)

    return pd.DataFrame(rows)


def build_numeric_summary(raw: pd.DataFrame) -> pd.DataFrame:
    g = (
        raw.groupby(
            ["Function", "Category", "Dimension", "Algorithm"],
            as_index=False,
        )
        .agg(
            Mean=("BestFitness", "mean"),
            Std=("BestFitness", "std"),
            Median=("BestFitness", "median"),
            Best=("BestFitness", "min"),
            MeanRuntimeSeconds=("RuntimeSeconds", "mean"),
            MeanFunctionEvaluations=("FunctionEvaluations", "mean"),
        )
    )
    return g


def paired_wilcoxon_vs_gfo(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for fname in complete_functions(raw):
        ref_df = (
            raw[
                (raw["Function"] == fname)
                & (raw["Algorithm"] == "GFO")
            ][["Run", "BestFitness"]]
            .rename(columns={"BestFitness": "GFO"})
        )

        for algo in ALGOS:
            if algo == "GFO":
                continue

            cmp_df = (
                raw[
                    (raw["Function"] == fname)
                    & (raw["Algorithm"] == algo)
                ][["Run", "BestFitness"]]
                .rename(columns={"BestFitness": algo})
            )

            paired = ref_df.merge(cmp_df, on="Run", how="inner").sort_values("Run")
            x = paired["GFO"].to_numpy(dtype=float)
            y = paired[algo].to_numpy(dtype=float)

            stat, p = safe_wilcoxon(x, y)

            mean_gfo = float(np.mean(x))
            mean_cmp = float(np.mean(y))

            if p < ALPHA:
                outcome = "Win" if mean_gfo < mean_cmp else "Loss"
            else:
                outcome = "Tie"

            rows.append(
                {
                    "Function": fname,
                    "ComparedAlgorithm": algo,
                    "GFO_Mean": mean_gfo,
                    "Competitor_Mean": mean_cmp,
                    "Statistic": stat,
                    "p_value": p,
                    "Outcome": outcome,
                }
            )

    df = pd.DataFrame(rows)

    # Holm correction separately for each competitor across benchmark functions.
    if not df.empty:
        df["p_holm"] = np.nan
        for algo in df["ComparedAlgorithm"].unique():
            idx = df.index[df["ComparedAlgorithm"] == algo].to_numpy()
            df.loc[idx, "p_holm"] = holm_adjust(
                df.loc[idx, "p_value"].to_numpy(dtype=float)
            )

    return df


def win_tie_loss_table(wilcox_df: pd.DataFrame, use_holm: bool = False) -> pd.DataFrame:
    rows = []

    for algo in [a for a in ALGOS if a != "GFO"]:
        sub = wilcox_df[wilcox_df["ComparedAlgorithm"] == algo].copy()

        if use_holm:
            # Reclassify according to Holm-adjusted significance.
            outcomes = []
            for _, row in sub.iterrows():
                if row["p_holm"] < ALPHA:
                    outcomes.append(
                        "Win"
                        if row["GFO_Mean"] < row["Competitor_Mean"]
                        else "Loss"
                    )
                else:
                    outcomes.append("Tie")
            sub["OutcomeHolm"] = outcomes
            col = "OutcomeHolm"
        else:
            col = "Outcome"

        rows.append(
            {
                "Competitor": algo,
                "Wins": int(np.sum(sub[col] == "Win")),
                "Ties": int(np.sum(sub[col] == "Tie")),
                "Losses": int(np.sum(sub[col] == "Loss")),
            }
        )

    return pd.DataFrame(rows)


def friedman_ranks(raw: pd.DataFrame, functions=None):
    if functions is None:
        functions = complete_functions(raw)

    functions = list(functions)

    if len(functions) < 2:
        return np.nan, np.nan, pd.DataFrame()

    M = np.zeros((len(functions), len(ALGOS)), dtype=float)

    for i, fname in enumerate(functions):
        for j, algo in enumerate(ALGOS):
            M[i, j] = raw.loc[
                (raw["Function"] == fname)
                & (raw["Algorithm"] == algo),
                "BestFitness",
            ].mean()

    ranks = np.vstack(
        [rankdata(M[i], method="average") for i in range(M.shape[0])]
    )
    avg = np.mean(ranks, axis=0)

    stat, p = friedmanchisquare(
        *[M[:, j] for j in range(M.shape[1])]
    )

    rank_df = pd.DataFrame(
        {
            "Algorithm": ALGOS,
            "AverageRank": avg,
        }
    ).sort_values("AverageRank").reset_index(drop=True)

    return float(stat), float(p), rank_df


def category_ranks(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for category in ["Unimodal", "Multimodal", "Fixed"]:
        funcs = [
            fname
            for fname, (_, _, _, _, cat) in BENCHMARKS.items()
            if cat == category
            and fname in complete_functions(raw)
        ]

        if len(funcs) < 2:
            continue

        stat, p, ranks = friedman_ranks(raw, funcs)

        for _, rr in ranks.iterrows():
            rows.append(
                {
                    "Category": category,
                    "N_Functions": len(funcs),
                    "Algorithm": rr["Algorithm"],
                    "AverageRank": rr["AverageRank"],
                    "FriedmanStatistic": stat,
                    "FriedmanP": p,
                }
            )

    return pd.DataFrame(rows)


def evaluation_summary(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby("Algorithm", as_index=False)
        .agg(
            MeanFunctionEvaluations=("FunctionEvaluations", "mean"),
            MedianFunctionEvaluations=("FunctionEvaluations", "median"),
            MeanRuntimeSeconds=("RuntimeSeconds", "mean"),
            MedianRuntimeSeconds=("RuntimeSeconds", "median"),
        )
        .sort_values("MeanFunctionEvaluations")
        .reset_index(drop=True)
    )


def save_convergence_summaries(raw: pd.DataFrame) -> None:
    """
    Average convergence histories across repetitions.
    Histories can have different lengths across algorithms, so each algorithm
    is summarized on its own iteration index.
    """
    rows = []

    for fname in BENCHMARKS:
        for algo in ALGOS:
            sub = raw[
                (raw["Function"] == fname)
                & (raw["Algorithm"] == algo)
            ]

            histories = []

            for _, row in sub.iterrows():
                path = Path(str(row["HistoryFile"]))
                if not path.exists():
                    continue

                with np.load(path, allow_pickle=False) as z:
                    histories.append(
                        np.asarray(z["history_best"], dtype=float)
                    )

            if not histories:
                continue

            max_len = max(len(h) for h in histories)

            # Pad shorter histories with their final best value.
            H = np.vstack(
                [
                    np.pad(
                        h,
                        (0, max_len - len(h)),
                        mode="edge",
                    )
                    for h in histories
                ]
            )

            mean_h = np.mean(H, axis=0)
            median_h = np.median(H, axis=0)

            for it in range(max_len):
                rows.append(
                    {
                        "Function": fname,
                        "Algorithm": algo,
                        "HistoryIndex": it,
                        "MeanBestFitness": mean_h[it],
                        "MedianBestFitness": median_h[it],
                        "N_Runs": H.shape[0],
                    }
                )

    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "continuous_convergence_summary.csv",
        index=False,
    )


def save_final_outputs(raw: pd.DataFrame) -> None:
    complete = complete_functions(raw)

    print()
    print("=" * 80)
    print("FINAL STATISTICAL ANALYSIS")
    print("=" * 80)
    print(f"Complete benchmark functions: {len(complete)}/{len(BENCHMARKS)}")

    mean_sd = build_mean_sd_table(raw)
    numeric = build_numeric_summary(raw)
    wilcox = paired_wilcoxon_vs_gfo(raw)
    wtl = win_tie_loss_table(wilcox, use_holm=False)
    wtl_holm = win_tie_loss_table(wilcox, use_holm=True)
    evals = evaluation_summary(raw)

    stat, p, ranks = friedman_ranks(raw, complete)
    cat_ranks = category_ranks(raw)

    mean_sd.to_csv(
        OUTPUT_DIR / "continuous_mean_sd_table.csv",
        index=False,
    )
    numeric.to_csv(
        OUTPUT_DIR / "continuous_numeric_summary.csv",
        index=False,
    )
    wilcox.to_csv(
        OUTPUT_DIR / "continuous_wilcoxon_signed_rank_vs_GFO.csv",
        index=False,
    )
    wtl.to_csv(
        OUTPUT_DIR / "continuous_win_tie_loss.csv",
        index=False,
    )
    wtl_holm.to_csv(
        OUTPUT_DIR / "continuous_win_tie_loss_holm.csv",
        index=False,
    )
    ranks.to_csv(
        OUTPUT_DIR / "continuous_friedman_ranks.csv",
        index=False,
    )
    cat_ranks.to_csv(
        OUTPUT_DIR / "continuous_category_ranks.csv",
        index=False,
    )
    evals.to_csv(
        OUTPUT_DIR / "continuous_evaluation_runtime_summary.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR / "continuous_friedman_test.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "friedman_stat": stat,
                "friedman_p": p,
                "n_complete_functions": len(complete),
                "R": R,
                "n_agents": N_AGENTS,
                "iterations": ITERS,
                "excluded_functions": sorted(EXCLUDED_FUNCTIONS),
            },
            f,
            indent=2,
        )

    save_convergence_summaries(raw)

    print("\nOverall Friedman ranks:")
    print(ranks.to_string(index=False))
    print(f"\nFriedman statistic = {stat:.6g}")
    print(f"Friedman p-value   = {p:.6g}")

    print("\nGFO win / tie / loss counts (raw paired Wilcoxon p-values):")
    print(wtl.to_string(index=False))

    print("\nGFO win / tie / loss counts (Holm-adjusted):")
    print(wtl_holm.to_string(index=False))

    print("\nMean objective-evaluation and runtime summary:")
    print(evals.to_string(index=False))

    print()
    print(f"All results saved in:\n{OUTPUT_DIR.resolve()}")


# ======================================================================
# MAIN
# ======================================================================

def main():
    print()
    print("=" * 80)
    print("GALACTIC FIELD OPTIMIZATION — FINAL CONTINUOUS STUDY")
    print("=" * 80)
    print(
        "This script is resumable. You may stop it and rerun it later; "
        "completed repetitions are retained."
    )
    print()
    print("Final benchmark decisions:")
    print("  - PermDBeta excluded pending separate numerical audit.")
    print("  - Alpine2 excluded because the raw standard form is ambiguous")
    print("    under minimization.")
    print(f"  - Trid domain set to [-D^2, D^2] = [{-(DEFAULT_D**2)}, {DEFAULT_D**2}].")
    print(f"  - Same population size ({N_AGENTS}) and iterations ({ITERS}) for all algorithms.")
    print("  - Objective-function evaluations are counted and reported.")
    print()

    raw = run_experiment()
    save_final_outputs(raw)

    print()
    print("=" * 80)
    print("RUN COMPLETE")
    print("=" * 80)
    print(
        "If PyCharm also reports 'Process finished with exit code 0', "
        "the complete experiment and final statistical tables were generated."
    )


if __name__ == "__main__":
    main()