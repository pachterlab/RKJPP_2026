"""Stress test: vary everything, and check nothing crosses the spectrum-free bound.

The claim the manuscript makes is that no model, at any cohort size, on any
genomic feature set, can extract more than I_n bits per patient. This script
tries to break it by sweeping the three axes an analyst actually controls:

  * data size      -- training-set size n, subsampled over a grid
  * genomics set   -- which genes enter (all HVGs, fewer HVGs, a curated anchor
                      panel, a random draw)
  * algorithm      -- ridge, RBF kernel ridge, random forest, gradient boosting,
                      PLS, and regularized CCA itself

Achieved information is measured on the same scale as the bound. Targets are the
top-k genomic canonical directions estimated on the *training* split (so they
are mutually uncorrelated with unit variance, exactly the structure I_n sums
over); each is predicted from the imaging PCs, and the held-out R^2 values are
combined as

    I_achieved = -1/2 sum_j log2(1 - max(R^2_j, 0))    bits.

Because that plug-in statistic has a positive floor under the null -- clipping
at zero and summing over several directions accumulates noise -- every
configuration also gets its own permutation null, and the reported value is
null-corrected. Comparing the raw statistic to the bound would risk a spurious
crossing that reflects estimator optimism rather than a real violation.

Usage:  python scripts/bound_stress_test.py [cohort ...]
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

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from rgit import fit_recoverability
from rgit.figures import label_panels
from rgit.bounds import max_information_given_fourth_moment
from attainable_bound_cohorts import (  # noqa: E402
    LOADERS, FIGDIR, OUTROOT, working_space, gaussian_rank, D_STAR, SEED,
)
from anchor_gene_saturation import PANELS  # noqa: E402

K_TARGETS = 5      # canonical directions predicted (bound covers all d*, so this
                   # comparison is conservative in the bound's favour)
N_REPS = 25
N_PERM = 25
_LOG2 = np.log(2.0)


def achieved_bits(r2_per_direction):
    r2 = np.clip(np.asarray(r2_per_direction, float), 0.0, 1.0 - 1e-9)
    return float(-0.5 * np.sum(np.log1p(-r2)) / _LOG2)


def algorithms(n_tr):
    zoo = {
        "ridge": (lambda: RidgeCV(alphas=np.logspace(-2, 5, 30)), True),
        "kernel ridge": (lambda: KernelRidge(kernel="rbf", alpha=1.0,
                                             gamma=1.0 / D_STAR), True),
        "random forest": (lambda: RandomForestRegressor(
            n_estimators=100, min_samples_leaf=5, random_state=0, n_jobs=-1), True),
        "grad. boosting": (lambda: HistGradientBoostingRegressor(
            max_iter=80, learning_rate=0.1, random_state=0), False),
        "PLS": (lambda: PLSRegression(n_components=min(5, D_STAR)), True),
    }
    return zoo


def genomic_sets(G_raw, sym, cohort):
    """The 'genomics set' axis: which genes the analyst chooses to look at."""
    rng = np.random.default_rng(SEED)
    var = G_raw.var(0)
    order = np.argsort(var)[::-1]
    sets = {
        "HVG 2000": order[:2000],
        "HVG 500": order[:500],
        "random 300": rng.choice(G_raw.shape[1], size=min(300, G_raw.shape[1]),
                                 replace=False),
    }
    sym_u = np.array([s.upper() for s in sym.astype(str)])
    idx = [int(np.flatnonzero(sym_u == g.upper())[0])
           for g in PANELS.get(cohort, []) if np.any(sym_u == g.upper())]
    if len(idx) >= 8:
        sets["anchor panel"] = np.array(idx)
    return sets


def one_config(Gw, Xw, n_grid, mk, multi, reps, seed, permute=False):
    """Mean achieved bits per training size for one (genomics set, algorithm)."""
    n = Gw.shape[0]
    rng = np.random.default_rng(seed)
    out = {}
    for ntr in n_grid:
        n_te = min(n - ntr, max(30, n // 4))
        if n_te < 15:
            continue
        vals = []
        K = int(min(K_TARGETS, Gw.shape[1], Xw.shape[1]))
        for _ in range(reps):
            perm = rng.permutation(n)
            tr, te = perm[:ntr], perm[ntr:ntr + n_te]
            G_use = Gw[rng.permutation(n)] if permute else Gw
            sc = StandardScaler().fit(Xw[tr])
            Xtr, Xte = sc.transform(Xw[tr]), sc.transform(Xw[te])
            try:
                fit = fit_recoverability(G_use[tr], Xw[tr], n_components=K)
                Ytr, Yte = fit.genomic_scores(G_use[tr]), fit.genomic_scores(G_use[te])
                sd = Ytr.std(0); sd[sd == 0] = 1.0
                Ytr, Yte = Ytr / sd, Yte / sd
                mu = Ytr.mean(0)
                if multi:
                    pred = mk().fit(Xtr, Ytr - mu).predict(Xte).reshape(Yte.shape) + mu
                else:
                    pred = np.column_stack([
                        mk().fit(Xtr, (Ytr - mu)[:, j]).predict(Xte) + mu[j]
                        for j in range(K)])
                num = np.sum((Yte - pred) ** 2, 0)
                den = np.sum((Yte - mu[None, :]) ** 2, 0)
                vals.append(achieved_bits(1.0 - num / den))
            except Exception:
                continue
        if vals:
            out[ntr] = np.asarray(vals, float)
    return out


def run(cohort):
    print(f"\n{'='*78}\n{cohort.upper()}: stress test against the spectrum-free bound\n{'='*78}")
    G_raw, X_raw, sym = LOADERS[cohort]()
    n = G_raw.shape[0]
    Xw = working_space(X_raw, D_STAR)
    d_star = Xw.shape[1]

    shp = json.load(open(OUTROOT / FIGDIR[cohort] / "shape_sensitivity.json"))
    S_bar = shp["S_bar_shape_swept"]
    n_grid = sorted({int(v) for v in np.round(
        np.geomspace(max(30, d_star + 10), int(n * 0.75), 6))})
    print(f"  n={n}, d*={d_star}, S_bar={S_bar:.3f}; training sizes {n_grid}")

    gsets = genomic_sets(G_raw, sym, cohort)
    print(f"  genomics sets: {', '.join(f'{k} ({len(v)} genes)' for k,v in gsets.items())}")

    results, nulls = {}, {}
    for gname, cols in gsets.items():
        Gw = working_space(G_raw[:, cols], D_STAR)
        for aname, (mk, multi) in algorithms(0).items():
            key = f"{gname} | {aname}"
            results[key] = one_config(Gw, Xw, n_grid, mk, multi, N_REPS, SEED)
            nulls[key] = one_config(Gw, Xw, n_grid, mk, multi, N_PERM, SEED + 99,
                                    permute=True)
            print(f"    {key:34s} done")

    ceiling = {ntr: max_information_given_fourth_moment(S_bar, ntr, d_star)
               for ntr in n_grid}

    print(f"\n  {'configuration':34s} " +
          " ".join(f"{v:>7d}" for v in n_grid))
    print(f"  {'CEILING (spectrum-free)':34s} " +
          " ".join(f"{ceiling[v]:7.3f}" for v in n_grid))
    # A crossing is only meaningful beyond Monte-Carlo error, so each cell
    # carries the s.e. of (mean achieved - mean null) and is tested at 2 s.e.
    corrected, errs, crossings = {}, {}, []
    for key in results:
        row, erow = {}, {}
        for ntr in n_grid:
            a, b = results[key].get(ntr), nulls[key].get(ntr)
            if a is None or b is None or len(a) < 2 or len(b) < 2:
                continue
            se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
            val = float(a.mean() - b.mean())
            row[ntr], erow[ntr] = max(val, 0.0), se
            if val - 2.0 * se > ceiling[ntr]:
                crossings.append({"config": key, "n": ntr, "value": val, "se": se,
                                  "ceiling": ceiling[ntr],
                                  "excess_sigma": (val - ceiling[ntr]) / se})
        corrected[key], errs[key] = row, erow
        print(f"  {key:34s} " +
              " ".join(f"{row.get(v, float('nan')):7.3f}" for v in n_grid))

    n_cells = sum(len(v) for v in corrected.values())
    print(f"\n  configurations x sizes tested: {n_cells}")
    print(f"  crossings beyond 2 s.e. (expected by chance: {0.023*n_cells:.1f}): "
          f"{len(crossings)}")
    for c in crossings:
        print("   ", c)
    nonzero = sum(1 for v in corrected.values()
                  if v and max(v.values()) > 0.01)
    print(f"  configurations reaching >0.01 bits at some n: {nonzero}/{len(corrected)}")

    res = {"cohort": cohort, "n": int(n), "d_star": int(d_star), "S_bar": S_bar,
           "n_grid": n_grid, "ceiling": ceiling,
           "achieved_raw": {k: {n: v.tolist() for n, v in d.items()}
                            for k, d in results.items()},
           "achieved_null_corrected": corrected, "se": errs,
           "crossings": crossings, "n_cells": n_cells,
           "expected_false_crossings": 0.023 * n_cells}
    (OUTROOT / FIGDIR[cohort] / "bound_stress_test.json").write_text(
        json.dumps(res, indent=2, default=float))
    return res


COHORT_LABEL = {"kirc": "TCGA-KIRC", "nsclc": "NSCLC", "adni": "ADNI"}


def figure(all_res):
    """Colour encodes the genomics set, marker/linestyle the algorithm."""
    cohorts = list(all_res)
    gcolors = {"HVG 2000": "tab:blue", "HVG 500": "tab:orange",
               "random 300": "tab:green", "anchor panel": "tab:red"}
    astyle = {"ridge": ("o", "-"), "kernel ridge": ("s", "--"),
              "random forest": ("^", "-."), "grad. boosting": ("D", ":"),
              "PLS": ("v", (0, (3, 1, 1, 1)))}

    fig, axes = plt.subplots(1, len(cohorts), figsize=(5.2 * len(cohorts), 4.6),
                             squeeze=False)
    for ax, c in zip(axes[0], cohorts):
        r = all_res[c]
        ng = [int(v) for v in r["n_grid"]]
        get = lambda d, k: d.get(k, d.get(str(k)))
        nd = np.geomspace(min(ng) * 0.9, max(ng) * 1.15, 200)
        ax.semilogx(nd, [max_information_given_fourth_moment(r["S_bar"], v, r["d_star"])
                         for v in nd], "k-", lw=3.0, zorder=10,
                    label="spectrum-free ceiling")
        for key, row in r["achieved_null_corrected"].items():
            gname, aname = [t.strip() for t in key.split("|")]
            xs = [v for v in ng if get(row, v) is not None]
            ys = [get(row, v) for v in xs]
            es = [get(r["se"].get(key, r["se"].get(str(key), {})), v) or 0.0
                  for v in xs]
            m, ls = astyle.get(aname, ("o", "-"))
            ax.errorbar(xs, ys, yerr=es, marker=m, ls=ls, ms=3.5, lw=1.0,
                        alpha=0.85, capsize=1.5, elinewidth=0.6,
                        color=gcolors.get(gname, "grey"))
        ax.axhline(0, color="grey", lw=0.8, ls=":")
        ax.set_xlim(min(ng) * 0.88, max(ng) * 1.18)
        top = max_information_given_fourth_moment(r["S_bar"], max(ng) * 1.15,
                                                  r["d_star"])
        ax.set_ylim(-0.03 * top, top * 1.06)
        ax.set_xlabel("training patients $n$")
        ax.set_ylabel("achieved information (bits/patient, null-corrected)")
        ax.grid(alpha=0.25)
        ax.text(0.03, 0.955,
                f"{COHORT_LABEL.get(c, c.upper())}, $n$ = {r['n']}\n"
                f"{len(r['achieved_null_corrected'])} configurations "
                f"$\\times$ {len(ng)} sizes\n"
                f"{len(r['crossings'])} crossings "
                f"({r['expected_false_crossings']:.1f} expected by chance)",
                transform=ax.transAxes, fontsize=7.8, va="top",
                bbox=dict(fc="white", ec="0.8", alpha=0.9, pad=3))

    handles = [plt.Line2D([], [], color="k", lw=3.0, label="spectrum-free ceiling")]
    handles += [plt.Line2D([], [], color=v, lw=2.2, label=k)
                for k, v in gcolors.items()]
    handles += [plt.Line2D([], [], color="0.35", marker=m, ls=ls, ms=4, lw=1.0,
                           label=k) for k, (m, ls) in astyle.items()]
    fig.legend(handles=handles, fontsize=8.5, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.16), frameon=False,
               title="black: ceiling   |   colour: genomics set   |   "
                     "marker: algorithm", title_fontsize=8.5)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    label_panels(fig, axes[0])
    fig.savefig(OUTROOT / "bound_stress_test.pdf", bbox_inches="tight")
    fig.savefig(OUTROOT / "bound_stress_test.png", dpi=160, bbox_inches="tight")
    print(f"\nwrote {OUTROOT/'bound_stress_test.pdf'}")


if __name__ == "__main__":
    if "--replot" in sys.argv:      # redraw from saved JSON, no recomputation
        figure(json.load(open(OUTROOT / "bound_stress_test.json")))
        sys.exit(0)
    out = {}
    for c in (sys.argv[1:] or ["kirc", "nsclc", "adni"]):
        try:
            out[c] = run(c)
        except Exception:
            import traceback
            traceback.print_exc()
    if out:
        figure(out)
        (OUTROOT / "bound_stress_test.json").write_text(
            json.dumps(out, indent=2, default=float))
        tot = sum(v["n_cells"] for v in out.values())
        cr = sum(len(v["crossings"]) for v in out.values())
        print(f"\nTOTAL: {tot} configuration x size cells, {cr} crossings")
