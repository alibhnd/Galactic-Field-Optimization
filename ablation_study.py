#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Component-level ablation study for Galactic Field Optimization (GFO).

This script reproduces the final six-variant ablation analysis reported in the
manuscript. It imports the final GFO mechanisms and 30-function benchmark suite
from gfo_continuous.py so that benchmark definitions and control parameters are
kept consistent with the main continuous study.

Reported ablation protocol
--------------------------
- 30 benchmark functions
- 30 agents
- 200 iterations
- 10 matched runs
- seeds 42 through 51
- variants:
    Full GFO
    w/o Orbital Motion
    w/o Adaptive Schedule
    w/o Diversification
    w/o Local Refinement
    w/o Restart

Run:
    python experiments/ablation_study.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

from gfo.continuous import (
    BENCHMARKS,
    GFOParams,
    clamp,
    compute_coordinate_scale,
    fitness_to_mass,
    gfo_schedule,
    leader_center,
    levy_flight,
    safe_norm,
    set_seed,
)

OUTPUT_DIR = Path("results/ablation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_AGENTS = 30
ITERS = 200
N_RUNS = 10
SEED0 = 42


@dataclass
class GFOAblationConfig:
    use_orbital: bool = True
    use_adaptive_schedule: bool = True
    use_diversification: bool = True
    use_local_refinement: bool = True
    use_restart: bool = True


ABLATION_VARIANTS = {
    "Full GFO": GFOAblationConfig(),
    "w/o Orbital Motion": GFOAblationConfig(use_orbital=False),
    "w/o Adaptive Schedule": GFOAblationConfig(use_adaptive_schedule=False),
    "w/o Diversification": GFOAblationConfig(use_diversification=False),
    "w/o Local Refinement": GFOAblationConfig(use_local_refinement=False),
    "w/o Restart": GFOAblationConfig(use_restart=False),
}


def orbital_update_ablation(X, F_tot, sch, lb, ub, coord_scale, cfg):
    n, d = X.shape
    F_dir = F_tot / safe_norm(F_tot, axis=1, keepdims=True)

    if cfg.use_orbital:
        U = np.random.randn(n, d)
        U = U / safe_norm(U, axis=1, keepdims=True)
        frac = float(sch["frac"])
        delta_eff = float(np.clip(sch["delta"] * (1.0 - frac) ** 2, 0.0, 1.0))
        d_raw = F_dir + delta_eff * U
        direction = d_raw / safe_norm(d_raw, axis=1, keepdims=True)
    else:
        direction = F_dir

    step = sch["s_base"] * (1.0 + sch["step_noise"] * np.random.randn(n))
    step = np.maximum(step, 0.0)
    X_new = X + step[:, None] * direction * coord_scale[None, :]
    return clamp(X_new, lb, ub)


def compute_field_ablation(X, m, center_x, sch, P):
    n, _ = X.shape
    Gt, lam = sch["G"], sch["lam"]

    Rij = X[None, :, :] - X[:, None, :]
    dij = safe_norm(Rij, axis=2, keepdims=True)
    dij_s = dij[..., 0]
    dir_ij = Rij / dij

    mi = m.reshape(-1, 1)
    mj = m.reshape(1, -1)
    fac_att = Gt * (mi * mj) / (dij_s**P.p + 1e-12)
    F_att = np.sum(fac_att[:, :, None] * dir_ij, axis=1)
    F_core = lam * (center_x[None, :] - X)
    return F_att + F_core


class GFOAblation:
    def __init__(self, func, lb, ub, params, cfg):
        self.func = func
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)
        self.p = params
        self.cfg = cfg
        self.d = self.lb.size
        self.delta_bar = float(np.mean(self.ub - self.lb))
        set_seed(self.p.seed)

        self.X = None
        self.f = None
        self.best_x = None
        self.best_f = np.inf
        self.history = []
        self._stag = 0
        self._boost_left = 0

    def _init_pop(self):
        n, d = self.p.n_agents, self.d
        self.X = self.lb + (self.ub - self.lb) * np.random.rand(n, d)
        self.f = np.asarray([self.func(x) for x in self.X], dtype=float)
        bi = int(np.argmin(self.f))
        self.best_x = self.X[bi].copy()
        self.best_f = float(self.f[bi])
        self.history = [self.best_f]

    def fit(self):
        self._init_pop()
        n, T, cfg = self.p.n_agents, self.p.iters, self.cfg

        for t in range(T):
            if cfg.use_adaptive_schedule:
                sch = gfo_schedule(t, T, self.delta_bar, self.p)
            else:
                t_mid = max(0, (T - 1) // 2)
                sch = gfo_schedule(t_mid, T, self.delta_bar, self.p)

            if cfg.use_restart and self._boost_left > 0:
                sch = dict(sch)
                sch["G"] *= self.p.boost_G_factor
                sch["delta"] = float(
                    np.clip(sch["delta"] * self.p.boost_delta_factor, 0.0, 1.0)
                )
                self._boost_left -= 1

            elite_ids = np.argsort(self.f)[:self.p.protected_elites]
            coord_scale = compute_coordinate_scale(self.X, elite_ids)
            m = fitness_to_mass(self.f, self.p.alpha_mass)
            lc = leader_center(self.X, self.f, weights=self.p.leader_weights)
            F_tot = compute_field_ablation(self.X, m, lc, sch, self.p)

            X_new = orbital_update_ablation(
                self.X, F_tot, sch, self.lb, self.ub, coord_scale, cfg
            )

            # Elite encircling retained in all variants.
            elite = self.best_x
            A = 2 * (1 - sch["frac"]) * np.random.rand(n, self.d) - (1 - sch["frac"])
            C = 2 * np.random.rand(n, self.d)
            D = np.abs(C * elite - self.X)
            X_encircle = elite - A * D
            if np.random.rand() >= (1 - sch["frac"] ** 0.5):
                X_new = X_encircle.copy()
            X_new = clamp(X_new, self.lb, self.ub)

            # Late exploitation retained in all variants.
            if sch["frac"] > 0.60:
                sigma = 0.20 * (1.0 - sch["frac"] ** 0.5)
                exploit_mask = np.random.rand(n) < 0.50
                direction = self.best_x - self.X
                X_exploit = (
                    self.X
                    + sigma * direction
                    + sigma * (self.ub - self.lb) * np.random.randn(n, self.d)
                )
                X_exploit = clamp(X_exploit, self.lb, self.ub)
                X_new[exploit_mask] = X_exploit[exploit_mask]

            # Early exploitation retained in all variants.
            if sch["frac"] < 0.25:
                strong_mask = np.random.rand(n) < 0.70
                X_strong = (
                    self.best_x
                    + 0.30 * (self.ub - self.lb) * np.random.randn(n, self.d)
                )
                X_strong = clamp(X_strong, self.lb, self.ub)
                X_new[strong_mask] = X_strong[strong_mask]

            if cfg.use_diversification:
                jump_mask = np.random.rand(n) < sch["jump_prob"]
                jump_mask[elite_ids] = False
                if np.any(jump_mask):
                    X_new[jump_mask] = self.lb + (self.ub - self.lb) * np.random.rand(
                        np.sum(jump_mask), self.d
                    )

                levy_mask = np.random.rand(n) < sch["levy_prob"]
                levy_mask[elite_ids] = False
                if np.any(levy_mask):
                    steps = levy_flight(np.sum(levy_mask), self.d)
                    levy_scale = 8.0 * self.p.levy_scale * (1.0 + sch["frac"])
                    X_new[levy_mask] = (
                        lc[None, :]
                        + levy_scale * steps * (self.ub - self.lb)[None, :]
                    )
                    X_new[levy_mask] = clamp(
                        X_new[levy_mask], self.lb, self.ub
                    )

            X_new = clamp(X_new, self.lb, self.ub)
            f_new = np.asarray([self.func(x) for x in X_new], dtype=float)

            improve = f_new < self.f
            self.X[improve] = X_new[improve]
            self.f[improve] = f_new[improve]

            bi = int(np.argmin(self.f))
            bf = float(self.f[bi])
            if bf < self.best_f - 1e-12:
                self.best_f = bf
                self.best_x = self.X[bi].copy()
                self._stag = 0
            else:
                self._stag += 1

            if cfg.use_local_refinement and sch["frac"] > self.p.local_search_start:
                refine_ids = np.argsort(self.f)[:self.p.elite_refine_count]
                for idx in refine_ids:
                    best_local_x = self.X[idx].copy()
                    best_local_f = float(self.f[idx])

                    for _ in range(self.p.refine_trials):
                        cand = (
                            best_local_x
                            + sch["local_sigma"]
                            * coord_scale
                            * (self.ub - self.lb)
                            * np.random.randn(self.d)
                        )
                        cand = clamp(cand, self.lb, self.ub)
                        f_cand = float(self.func(cand))
                        if f_cand < best_local_f:
                            best_local_x = cand
                            best_local_f = f_cand

                    self.X[idx] = best_local_x
                    self.f[idx] = best_local_f
                    if best_local_f < self.best_f:
                        self.best_f = best_local_f
                        self.best_x = best_local_x.copy()
                        self._stag = 0

            if cfg.use_restart and self._stag >= self.p.stagnation_window:
                self._stag = 0
                self._boost_left = self.p.boost_iters
                restart_frac = min(
                    0.50, self.p.restart_frac + 0.30 * sch["frac"]
                )
                k = max(1, int(restart_frac * n))
                worst_ids = np.argsort(self.f)[-k:]

                self.X[worst_ids] = (
                    self.lb
                    + (self.ub - self.lb) * np.random.rand(k, self.d)
                )
                self.f[worst_ids] = np.asarray(
                    [self.func(x) for x in self.X[worst_ids]], dtype=float
                )
                bi2 = int(np.argmin(self.f))
                if float(self.f[bi2]) < self.best_f:
                    self.best_f = float(self.f[bi2])
                    self.best_x = self.X[bi2].copy()

            self.history.append(self.best_f)

        return self.best_x, self.best_f, {
            "history_best": np.asarray(self.history, dtype=float)
        }


def safe_wilcoxon(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.all(np.isclose(x - y, 0.0, rtol=1e-12, atol=1e-15)):
        return 0.0, 1.0
    try:
        return tuple(
            float(v)
            for v in wilcoxon(
                x, y, zero_method="wilcox", alternative="two-sided", method="auto"
            )
        )
    except ValueError:
        return np.nan, 1.0


def run_ablation():
    variants = list(ABLATION_VARIANTS)
    records = []

    total = len(BENCHMARKS) * len(variants) * N_RUNS
    done = 0

    for fname, (func, lb, ub, dim, category) in BENCHMARKS.items():
        lbv = np.full(dim, lb, dtype=float)
        ubv = np.full(dim, ub, dtype=float)

        for variant in variants:
            for run in range(N_RUNS):
                seed = SEED0 + run
                P = GFOParams(n_agents=N_AGENTS, iters=ITERS, seed=seed)
                model = GFOAblation(
                    func, lbv, ubv, params=P, cfg=ABLATION_VARIANTS[variant]
                )
                _, best_f, _ = model.fit()

                records.append({
                    "Function": fname,
                    "Category": category,
                    "Dimension": dim,
                    "Variant": variant,
                    "Run": run + 1,
                    "Seed": seed,
                    "BestFitness": float(best_f),
                })
                done += 1
                print(f"[{done:4d}/{total}] {fname} | {variant} | run {run+1}")

    raw = pd.DataFrame(records)
    raw.to_csv(OUTPUT_DIR / "ablation_raw_runs.csv", index=False)

    means = (
        raw.groupby(["Function", "Variant"], as_index=False)["BestFitness"]
        .mean()
        .pivot(index="Function", columns="Variant", values="BestFitness")
        .dropna(subset=variants)
    )

    rank_matrix = np.vstack([
        rankdata(means.loc[f, variants].to_numpy(dtype=float), method="average")
        for f in means.index
    ])
    rank_df = pd.DataFrame({
        "Variant": variants,
        "Average_Rank": rank_matrix.mean(axis=0),
    }).sort_values("Average_Rank")
    rank_df.to_csv(OUTPUT_DIR / "ablation_friedman_ranks.csv", index=False)

    stat, p = friedmanchisquare(
        *[means[v].to_numpy(dtype=float) for v in variants]
    )
    with open(OUTPUT_DIR / "ablation_friedman_test.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "statistic": float(stat),
                "p_value": float(p),
                "n_functions": int(len(means)),
                "n_runs": N_RUNS,
                "n_agents": N_AGENTS,
                "iterations": ITERS,
                "seed0": SEED0,
            },
            f,
            indent=2,
        )

    # Matched-run paired Wilcoxon comparisons against Full GFO, by function.
    pair_rows = []
    for fname in means.index:
        d = raw[raw["Function"] == fname]
        ref = (
            d[d["Variant"] == "Full GFO"][["Run", "BestFitness"]]
            .rename(columns={"BestFitness": "Full GFO"})
        )
        for variant in variants:
            if variant == "Full GFO":
                continue
            comp = (
                d[d["Variant"] == variant][["Run", "BestFitness"]]
                .rename(columns={"BestFitness": variant})
            )
            paired = ref.merge(comp, on="Run").sort_values("Run")
            W, pv = safe_wilcoxon(
                paired["Full GFO"].to_numpy(dtype=float),
                paired[variant].to_numpy(dtype=float),
            )
            pair_rows.append({
                "Function": fname,
                "Comparison": f"Full GFO vs {variant}",
                "W": W,
                "p_value": pv,
            })

    pd.DataFrame(pair_rows).to_csv(
        OUTPUT_DIR / "ablation_paired_wilcoxon_vs_full.csv", index=False
    )

    summary = (
        raw.groupby(["Function", "Variant"])["BestFitness"]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.to_csv(OUTPUT_DIR / "ablation_mean_sd.csv", index=False)

    print("\nOverall Friedman ranks:")
    print(rank_df.to_string(index=False))
    print(f"\nFriedman statistic = {stat:.6g}")
    print(f"Friedman p-value   = {p:.6g}")
    print(f"\nResults saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    run_ablation()
