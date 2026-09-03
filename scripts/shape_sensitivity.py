"""Does the S-inversion depend on the spectrum's shape, or only on S = sum rho_i^4?

The true bound  I_n <= d* g(sqrt(S/d*))  converts a fourth-moment budget S into
an information limit exactly, over every spectrum. What is still estimated from
a restricted family is S itself: the calibration plants flat rank-r spectra, so
S_bar is a supremum over flat shapes.

This script plants NON-flat spectra (geometric decay at several rates) with S
matched to a flat cell and compares the 5th percentile of the plug-in statistic.
If q05 depends only weakly on shape at matched S, the S-inversion is
approximately shape-free and the residual gap is small. If some shape gives a
markedly SMALLER q05, that shape is less detectable at the same S and the
inversion should use it -- so we report the minimum.

Usage:  python scripts/shape_sensitivity.py [cohort ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from rgit.bounds import attainable_recoverability
from attainable_bound_cohorts import (  # noqa: E402
    LOADERS, FIGDIR, OUTROOT, working_space, N_HVG, D_STAR, SEED,
)
from total_information_ucl import (  # noqa: E402
    info_stat, solve_t, block_dirs, N_FOLDS, _LOG2,
)

N_BOOT = 60
# fourth-moment budgets to probe, spanning the consistent region
S_TARGETS = [0.02, 0.05, 0.10, 0.20, 0.29, 0.42, 0.60, 0.85, 1.20]
# shapes: geometric decay rho_i^2 ∝ q^(i-1), q = 1 is flat
DECAYS = {"flat": 1.0, "mild": 0.7, "steep": 0.4, "v.steep": 0.2}
RANKS_PLANT = [8, 20]   # nonzero directions in the planted spectrum


def spectrum_for(S, q, r, cap=0.95):
    """Geometric spectrum with r terms and sum_i (rho_i^2)^2 = S."""
    w = q ** np.arange(r)
    scale = np.sqrt(S / np.sum(w ** 2))
    spec = scale * w
    return spec if spec.max() <= cap else None


def true_bits(spec, n, d_star):
    return float(sum(-0.5 * np.log1p(-attainable_recoverability(u, n, d_star)[0])
                     for u in spec) / _LOG2)


def plant(G, X, spec, n, d_star, Sgi, Sxi, rng):
    r = len(spec)
    U, W = block_dirs(G.shape[1], r, rng), block_dirs(d_star, r, rng)
    if U is None or W is None:
        return None
    Gp, Xp = G.copy(), X[rng.permutation(n)].copy()
    for u, w, rho2 in zip(U, W, spec):
        A, B = float(u @ Sgi @ u), float(w @ Sxi @ w)
        al = np.sqrt(solve_t(np.sqrt(rho2), A, B))
        Z = rng.standard_normal(n)
        Gp += al * np.outer(Z, u)
        Xp += al * np.outer(Z, w)
    return info_stat(Gp, Xp, n, d_star)


def leading_cap(name):
    """rho_1^2 upper limit from the rank-one calibration (channel_ucl.py)."""
    j = json.load(open(OUTROOT / FIGDIR[name] / "channel_ucl.json"))
    return float(j["channel_rho2_ucl95"])


def run(name):
    print(f"\n{'='*76}\n{name.upper()}: shape sensitivity of the S-inversion\n{'='*76}")
    G_raw, X_raw, _ = LOADERS[name]()
    n = G_raw.shape[0]
    hv = np.argsort(G_raw.var(0))[::-1][:N_HVG]
    G = working_space(G_raw[:, hv], D_STAR)
    X = working_space(X_raw, D_STAR)
    d_star = X.shape[1]
    Sgi = np.linalg.inv(np.cov(G.T) + 1e-8 * np.eye(G.shape[1]))
    Sxi = np.linalg.inv(np.cov(X.T) + 1e-8 * np.eye(d_star))
    obs = info_stat(G, X, n, d_star)
    cap = leading_cap(name)
    rng = np.random.default_rng(SEED)

    print(f"  n={n}, d*={d_star}, observed = {obs:.3f} bits")
    print(f"  spectra are additionally required to satisfy rho_1^2 <= {cap:.3f} "
          f"(leading-axis UCL), which is what rules out steep shapes at large S")
    print(f"\n  {'S':>6s} {'rank':>5s} " + " ".join(f"{k:>13s}" for k in DECAYS))
    rows = []
    for S in S_TARGETS:
      for RANK in RANKS_PLANT:
        q05s, cells = {}, {}
        for label, q in DECAYS.items():
            spec = spectrum_for(S, q, RANK, cap=cap)
            if spec is None:
                q05s[label] = np.nan   # violates the leading-axis UCL
                continue
            vals = [plant(G, X, spec, n, d_star, Sgi, Sxi, rng) for _ in range(N_BOOT)]
            vals = np.array([v for v in vals if v is not None])
            q05s[label] = float(np.quantile(vals, 0.05))
            cells[label] = {"q05": q05s[label], "true_bits": true_bits(spec, n, d_star),
                            "consistent": bool(q05s[label] <= obs)}
        fin = [v for v in q05s.values() if np.isfinite(v)]
        spread = (max(fin) - min(fin)) if fin else np.nan
        marks = []
        for label in DECAYS:
            v = q05s[label]
            marks.append("     cap-out" if not np.isfinite(v)
                         else f"{v:8.4f} {'ok' if v <= obs else 'XX'}")
        print(f"  {S:6.3f} {RANK:5d} " + " ".join(f"{m:>13s}" for m in marks))
        rows.append({"S": S, "rank": RANK, "q05": q05s, "cells": cells,
                     "spread": spread})

    # S_bar under the least favourable shape at each S
    consistent_S = [r["S"] for r in rows
                    if any(np.isfinite(v) and v <= obs for v in r["q05"].values())]
    S_bar = max(consistent_S, default=0.0)
    u = np.sqrt(S_bar / d_star)
    true_bound = d_star * float(
        -0.5 * np.log1p(-attainable_recoverability(u, n, d_star)[0]) / _LOG2) if u < 1 else np.inf
    print(f"\n  S_bar (largest S with ANY admissible shape consistent) = {S_bar:.4f}")
    print(f"  TRUE bound  d* g(sqrt(S_bar/d*))              = {true_bound:.3f} bits")

    res = {"cohort": name, "n": int(n), "d_star": int(d_star), "observed_bits": obs,
           "leading_cap_rho2": cap, "ranks_planted": RANKS_PLANT, "rows": rows,
           "S_bar_shape_swept": S_bar, "true_bound_bits": true_bound}
    (OUTROOT / FIGDIR[name] / "shape_sensitivity.json").write_text(
        json.dumps(res, indent=2, default=float))
    return res


if __name__ == "__main__":
    out = {}
    for c in (sys.argv[1:] or ["kirc", "nsclc", "adni"]):
        try:
            out[c] = run(c)
        except Exception:
            import traceback
            traceback.print_exc()
    (OUTROOT / "shape_sensitivity.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {OUTROOT/'shape_sensitivity.json'}")
