"""Upper confidence limit on TOTAL attainable information, not just the leading axis.

`channel_ucl.py` plants a rank-one signal and inverts on the leading held-out
recoverability, so it bounds rho_1^2 -- the maximum over canonical axes. Summing
over directions, the leading-axis figure is a *lower* bound on total retention,
which is the wrong direction for a headline claim.

Here the planted signal has rank r with equal per-direction strength, and the
statistic is the plug-in attainable information itself,

    I_hat = -1/2 sum_i log2(1 - max(R_i^cv, 0)),

so the inversion returns a limit on the quantity of interest directly:

    I_bar = max { r * g(rho^2) : q05(I_hat | r, rho^2) <= I_hat_obs },
    g(u)  = -1/2 log2(1 - R_n(u)).

Taking the maximum over the (r, rho^2) family is what makes this an upper bound
rather than a point estimate: it is the most information any spectrum in the
family could carry while still being consistent, at 5%, with what was observed.

Planting is exact. Both modalities are PCA working spaces, so their covariances
are diagonal; putting the r planted directions on disjoint blocks of principal
components makes the sub-channels mutually orthogonal, and each one's population
canonical correlation follows from the rank-one Sherman--Morrison solve.

Usage:  python scripts/total_information_ucl.py [cohort ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rgit import cross_validated_recoverability
from rgit.bounds import (
    attainable_information, attainable_recoverability, max_total_information,
)
from attainable_bound_cohorts import (  # noqa: E402
    LOADERS, FIGDIR, OUTROOT, working_space, N_HVG, D_STAR, SEED,
)

RANKS = [1, 2, 3, 5, 8, 12, 20]   # must reach d*, or the supremum is truncated
GRID = np.array([0.0, 0.02, 0.05, 0.09, 0.14, 0.20, 0.28, 0.38, 0.50, 0.65])
N_BOOT = 60
N_FOLDS = 5
_LOG2 = np.log(2.0)


def info_stat(G, X, n, d_star, k=None):
    """Plug-in attainable information (bits) from the held-out spectrum."""
    k = int(k or min(G.shape[1], X.shape[1]))
    R = np.clip(
        cross_validated_recoverability(G, X, k, N_FOLDS, random_state=SEED).mean(0),
        0.0, 1.0 - 1e-9)
    return float(-0.5 * np.sum(np.log1p(-R)) / _LOG2)


def rho_of_t(t, A, B):
    return t * np.sqrt(A * B) / np.sqrt((1 + t * A) * (1 + t * B))


def solve_t(target_rho, A, B):
    lo, hi = 0.0, 1.0
    while rho_of_t(hi, A, B) < target_rho and hi < 1e12:
        hi *= 2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if rho_of_t(mid, A, B) < target_rho:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def block_dirs(dim, r, rng):
    """r unit vectors on disjoint blocks of coordinates (hence Sigma-orthogonal)."""
    edges = np.linspace(0, dim, r + 1).astype(int)
    dirs = []
    for j in range(r):
        v = np.zeros(dim)
        lo, hi = edges[j], edges[j + 1]
        if hi <= lo:
            return None
        v[lo:hi] = rng.standard_normal(hi - lo)
        v /= np.linalg.norm(v)
        dirs.append(v)
    return dirs


def calibrate(G, X, n, d_star, seed=SEED):
    p_star = G.shape[1]
    Sgi = np.linalg.inv(np.cov(G.T) + 1e-8 * np.eye(p_star))
    Sxi = np.linalg.inv(np.cov(X.T) + 1e-8 * np.eye(d_star))
    rng = np.random.default_rng(seed)
    cells = []

    for r in RANKS:
        for r2 in GRID:
            target = np.sqrt(r2)
            vals = []
            for _ in range(N_BOOT):
                U = block_dirs(p_star, r, rng)
                W = block_dirs(d_star, r, rng)
                if U is None or W is None:
                    break
                Gp, Xp = G.copy(), X[rng.permutation(n)].copy()
                for u, w in zip(U, W):
                    A = float(u @ Sgi @ u)
                    B = float(w @ Sxi @ w)
                    al = np.sqrt(solve_t(target, A, B))
                    Z = rng.standard_normal(n)
                    Gp += al * np.outer(Z, u)
                    Xp += al * np.outer(Z, w)
                vals.append(info_stat(Gp, Xp, n, d_star))
            if not vals:
                continue
            vals = np.asarray(vals)
            cells.append({
                "rank": r, "rho2": float(r2),
                "true_bits": r * float(
                    -0.5 * np.log1p(-attainable_recoverability(r2, n, d_star)[0]) / _LOG2),
                "q05": float(np.quantile(vals, 0.05)),
                "median": float(np.median(vals)),
            })
    return cells


def run(name):
    print(f"\n{'='*72}\n{name.upper()}: TOTAL attainable-information UCL\n{'='*72}")
    G_raw, X_raw, _ = LOADERS[name]()
    n = G_raw.shape[0]
    hv = np.argsort(G_raw.var(0))[::-1][:N_HVG]
    G = working_space(G_raw[:, hv], D_STAR)
    X = working_space(X_raw, D_STAR)
    d_star = X.shape[1]

    obs = info_stat(G, X, n, d_star)
    cells = calibrate(G, X, n, d_star)

    null_cells = [c for c in cells if c["rho2"] == 0.0]
    null_med = float(np.median([c["median"] for c in null_cells])) if null_cells else float("nan")
    consistent = [c for c in cells if c["q05"] <= obs]
    ucl = max((c["true_bits"] for c in consistent), default=0.0)
    arg = max(consistent, key=lambda c: c["true_bits"]) if consistent else None

    # the leading-axis figure previously reported, for comparison
    lead = json.load(open(OUTROOT / FIGDIR[name] / "channel_ucl.json"))
    lead_bits = lead["attainable_information_bits_ucl"]
    lead_rho2 = lead["channel_rho2_ucl95"]

    print(f"  n = {n}, d* = {d_star}")
    print(f"  observed plug-in attainable information  = {obs:.3f} bits "
          f"(sum over {d_star} held-out directions)")
    print(f"  permutation null (rho^2 = 0) median      = {null_med:.3f} bits "
          f"-- the statistic's own floor; the observed value is "
          f"{'ABOVE' if obs > null_med else 'AT OR BELOW'} it")
    print(f"  leading-axis UCL (channel_ucl.py)        = {lead_bits:.3f} bits "
          f"at rho_1^2 <= {lead_rho2:.3f}")
    print(f"  TOTAL attainable-information UCL         = {ucl:.3f} bits", end="")
    if arg:
        print(f"  (worst case: rank {arg['rank']}, rho^2 = {arg['rho2']:.2f})")
    else:
        print()
    print(f"  concentration bound at the leading UCL   = "
          f"{max_total_information(lead_rho2, n, d_star, rho2_max=lead_rho2):.3f} bits")
    print(f"\n  {'rank':>5s} " + " ".join(f"{g:6.2f}" for g in GRID))
    for r in RANKS:
        row = [c for c in cells if c["rank"] == r]
        if not row:
            continue
        marks = []
        for g in GRID:
            c = next((x for x in row if abs(x["rho2"] - g) < 1e-9), None)
            marks.append("  --  " if c is None
                         else (f"{c['true_bits']:6.2f}" if c["q05"] <= obs else "  x   "))
        print(f"  {r:5d} " + " ".join(marks))
    print("  (values = true total bits of that spectrum; 'x' = excluded at 5%)")

    res = {
        "cohort": name, "n": int(n), "d_star": int(d_star),
        "observed_plugin_bits": obs,
        "null_median_bits": null_med,
        "total_information_ucl_bits": float(ucl),
        "argmax_cell": arg,
        "leading_axis_ucl_bits": lead_bits,
        "leading_axis_rho2_ucl": lead_rho2,
        "ranks": RANKS, "grid_rho2": GRID.tolist(), "cells": cells,
    }
    figdir = OUTROOT / FIGDIR[name]
    (figdir / "total_information_ucl.json").write_text(json.dumps(res, indent=2))

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for r in RANKS:
        row = sorted([c for c in cells if c["rank"] == r], key=lambda c: c["rho2"])
        if not row:
            continue
        ax.plot([c["true_bits"] for c in row], [c["q05"] for c in row],
                "o-", ms=3.5, lw=1.2, label=f"planted rank {r}")
    ax.axhline(obs, color="crimson", ls="--", lw=1.6,
               label=f"observed = {obs:.2f} bits")
    ax.axvline(ucl, color="k", ls=":", lw=1.8, label=f"total UCL = {ucl:.2f} bits")
    ax.axvline(lead_bits, color="grey", ls="-.", lw=1.3,
               label=f"leading-axis UCL = {lead_bits:.2f} bits")
    ax.set_xlabel("true total attainable information of the planted spectrum (bits)")
    ax.set_ylabel(r"5th percentile of the plug-in statistic")
    ax.set_xscale("log")
    ax.set_title(f"{name.upper()}: inverting on total information (n={n})")
    ax.legend(fontsize=7.2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figdir / "total_information_ucl.pdf", bbox_inches="tight")
    fig.savefig(figdir / "total_information_ucl.png", dpi=160, bbox_inches="tight")
    print(f"\n  wrote {figdir/'total_information_ucl.pdf'}")
    return res


if __name__ == "__main__":
    which = sys.argv[1:] or ["kirc", "nsclc", "adni"]
    out = {}
    for c in which:
        try:
            out[c] = run(c)
        except Exception:
            import traceback
            traceback.print_exc()
    (OUTROOT / "total_information_ucl.json").write_text(
        json.dumps(out, indent=2, default=float))
    print(f"\nwrote {OUTROOT/'total_information_ucl.json'}")
