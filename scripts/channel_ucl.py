"""Model-free upper confidence limit on the channel recoverability, and a
direct empirical check of the closed-form attainable ceiling.

Two things happen here.

1.  **Calibration.** Signal of *known* population strength is planted into
    permuted real data:

        G' = G + a Z u^T,   X' = X_perm + a Z w^T,   Z ~ N(0,1),

    so the cross-covariance is exactly rank one and the population canonical
    correlation follows in closed form by Sherman--Morrison. Permuting X first
    means that at rho^2 = 0 the reference distribution *is* the permutation
    null, and each modality keeps its own covariance and tail structure. This is
    the semi-parametric calibration of the manuscript's UCL section.

2.  **Two readings of the calibration curve.**
    - Inverting it at the observed statistic gives a one-sided 95% upper
      confidence limit on the *channel* value rho^2, hence (by monotonicity of
      R_n in rho^2) an upper limit on the attainable ceiling that owes nothing
      to any trained model.
    - Its *median* is a direct, assumption-light estimate of the attainable
      recoverability at the planted channel strength -- exactly what
      Proposition "attainable-recoverability ceiling" predicts in closed form.
      Plotting the two against each other tests the theory on real data.

Usage:  python scripts/channel_ucl.py [cohort ...]
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
    attainable_recoverability,
    attainable_information,
    channel_information,
    learning_cost,
    sample_size_for_fraction,
)
from attainable_bound_cohorts import (  # noqa: E402
    LOADERS, FIGDIR, OUTROOT, working_space, N_HVG, D_STAR, SEED,
)

GRID = np.array([0.0, 0.02, 0.05, 0.09, 0.14, 0.20, 0.28, 0.38, 0.50, 0.65])
N_BOOT = 120        # planted replicates per grid point
N_DIRS = 12         # random (u, w) direction pairs per grid point
N_FOLDS = 5


def stat(G, X):
    """The observed estimand: leading held-out (5-fold) recoverability."""
    v = cross_validated_recoverability(G, X, 1, N_FOLDS, random_state=SEED).mean(0)
    return float(max(v[0], 0.0))


def rho_of_t(t, A, B):
    """Population canonical correlation of the rank-one planted channel."""
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


def calibrate(G, X, seed=SEED):
    """Distribution of the statistic as a function of the planted channel rho^2."""
    n, p_star = G.shape
    d_star = X.shape[1]
    Sgi = np.linalg.inv(np.cov(G.T) + 1e-8 * np.eye(p_star))
    Sxi = np.linalg.inv(np.cov(X.T) + 1e-8 * np.eye(d_star))
    rng = np.random.default_rng(seed)

    q05, med, q95 = [], [], []
    for r2 in GRID:
        target = np.sqrt(r2)
        vals = []
        for _ in range(N_DIRS):
            u = rng.standard_normal(p_star); u /= np.linalg.norm(u)
            w = rng.standard_normal(d_star); w /= np.linalg.norm(w)
            A = float(u @ Sgi @ u)
            B = float(w @ Sxi @ w)
            t = 0.0 if r2 == 0 else solve_t(target, A, B)
            al = np.sqrt(t)
            for _ in range(max(N_BOOT // N_DIRS, 1)):
                Xp = X[rng.permutation(n)]
                Z = rng.standard_normal(n)
                vals.append(stat(G + al * np.outer(Z, u), Xp + al * np.outer(Z, w)))
        vals = np.asarray(vals)
        q05.append(float(np.quantile(vals, 0.05)))
        med.append(float(np.median(vals)))
        q95.append(float(np.quantile(vals, 0.95)))
    return np.array(q05), np.array(med), np.array(q95)


def invert(obs, q05):
    """UCL = sup{rho^2 : q05(stat | rho^2) <= obs}, by interpolation."""
    if obs < q05[0]:
        return 0.0
    if obs >= q05[-1]:
        return float(GRID[-1])
    return float(np.interp(obs, q05, GRID))


def run(name):
    print(f"\n{'='*72}\n{name.upper()}: channel UCL and ceiling validation\n{'='*72}")
    G_raw, X_raw, _ = LOADERS[name]()
    n = G_raw.shape[0]
    hv = np.argsort(G_raw.var(0))[::-1][:N_HVG]
    G = working_space(G_raw[:, hv], D_STAR)
    X = working_space(X_raw, D_STAR)
    d_star = X.shape[1]

    obs = stat(G, X)
    q05, med, q95 = calibrate(G, X)
    ucl = invert(obs, q05)

    # closed-form prediction for the same planted channel strengths
    theory = np.array([attainable_recoverability(r, n, d_star)[0] for r in GRID])
    # theory is for a fully-known target direction; the statistic also pays for
    # estimating it, so theory should sit at or above the calibration median.
    above = float(np.mean(theory >= med - 1e-12))

    R_ucl = float(attainable_recoverability(ucl, n, d_star)[0])
    print(f"  n = {n},  p* = {G.shape[1]},  d* = {d_star}")
    print(f"  observed held-out R_1              = {obs:.4f}")
    print(f"  permutation null (rho^2 = 0): median {med[0]:.4f}, q95 {q95[0]:.4f}")
    print(f"  one-sided 95% UCL on channel rho^2 = {ucl:.4f}")
    print(f"  => attainable ceiling at n={n}     = {R_ucl:.4f}")
    print(f"  => learning cost nu                = {learning_cost(ucl, d_star)[0]:.0f}"
          if ucl > 0 else "  => learning cost nu                = inf")
    print(f"\n  closed-form ceiling vs calibration median (planted rho^2):")
    print(f"    {'rho^2':>7s} {'theory R_n':>11s} {'calib median':>13s} {'calib q05':>10s}")
    for r, t, m, q in zip(GRID, theory, med, q05):
        print(f"    {r:7.2f} {t:11.4f} {m:13.4f} {q:10.4f}")
    print(f"  closed-form ceiling >= calibration median at "
          f"{100*above:.0f}% of planted values")

    res = {
        "cohort": name, "n": int(n), "d_star": int(d_star),
        "observed_R1_cv": obs,
        "grid_rho2": GRID.tolist(),
        "calib_q05": q05.tolist(), "calib_median": med.tolist(),
        "calib_q95": q95.tolist(),
        "theory_R_n": theory.tolist(),
        "theory_dominates_fraction": above,
        "channel_rho2_ucl95": ucl,
        "attainable_R_at_n_ucl": R_ucl,
        "attainable_information_bits_ucl": attainable_information([ucl], n, d_star),
        "channel_information_bits_ucl": channel_information([ucl]),
        "n_for_90pct_at_ucl": (float(sample_size_for_fraction(ucl, d_star, 0.9)[0])
                               if ucl > 0 else float("inf")),
        "perm_null_median": float(med[0]), "perm_null_q95": float(q95[0]),
    }

    figdir = OUTROOT / FIGDIR[name]
    figdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.fill_between(GRID, q05, q95, alpha=0.2, color="tab:blue",
                    label="planted-signal calibration (5--95%)")
    ax.plot(GRID, med, "o-", color="tab:blue", ms=4, label="calibration median")
    ax.plot(GRID, theory, "k-", lw=2.2,
            label=r"closed-form ceiling $\mathcal{R}_n$")
    ax.axhline(obs, color="crimson", ls="--", lw=1.5,
               label=f"observed $\\hat R_1^{{cv}}$ = {obs:.3f}")
    ax.axvline(ucl, color="crimson", ls=":", lw=1.5,
               label=rf"95% UCL on $\rho^2$ = {ucl:.3f}")
    ax.set_xlabel(r"planted channel recoverability $\rho^2$")
    ax.set_ylabel(r"held-out $\hat R_1^{\mathrm{cv}}$")
    ax.set_title(f"{name.upper()}: channel UCL by test inversion (n={n})")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figdir / "channel_ucl.pdf", bbox_inches="tight")
    fig.savefig(figdir / "channel_ucl.png", dpi=160, bbox_inches="tight")
    (figdir / "channel_ucl.json").write_text(json.dumps(res, indent=2))
    print(f"\n  wrote {figdir/'channel_ucl.pdf'}")
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
    (OUTROOT / "channel_ucl.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {OUTROOT/'channel_ucl.json'}")
