"""Attainable-information ceiling on the three real radiogenomic cohorts.

For each cohort we (i) fix an imaging working dimension d*, (ii) take as
prediction target the leading genomic canonical directions estimated on the
training split alone (nested CCA -- no test patient enters the target's
construction), (iii) sweep the training-set size, training several model
families at each size and scoring them on held-out patients, and (iv) fit the
one-parameter ceiling

    R_n(rho^2) = n rho^4 / (d* + rho^2 (n - d*))

to the resulting learning curve. The fitted rho^2 is the extrapolated *channel*
recoverability (n -> infinity); the curve itself is the finite-n ceiling that no
trained model should cross.

Usage:  python scripts/attainable_bound_cohorts.py [cohort ...]
        (default: all of kirc nsclc adni)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import scipy.sparse as sp
import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import minimize_scalar
from scipy.stats import rankdata
from scipy.special import ndtri
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler

from rgit import fit_recoverability, cross_validated_recoverability
from rgit.bounds import (
    attainable_recoverability,
    attainable_information,
    channel_information,
    learning_cost,
    sample_size_for_fraction,
)

REPO = Path(__file__).parent.parent
OUTROOT = REPO / "notebooks" / "figures"

D_STAR = 20        # imaging working dimension, held FIXED across the sweep
P_TARGETS = 3      # canonical directions used as prediction targets
N_REPS = 30        # random splits per training size
N_HVG = 2000
SEED = 0


# ---------------------------------------------------------------------------
# preprocessing (same conventions as scripts/recoverability_diagnostics.py)
# ---------------------------------------------------------------------------
def dense(M):
    return M.toarray() if sp.issparse(M) else np.asarray(M)


def gaussian_rank(M):
    M = np.asarray(M, dtype=np.float64)
    n = M.shape[0]
    R = np.apply_along_axis(lambda c: rankdata(c, method="average"), 0, M)
    return ndtri(R / (n + 1.0))


def untied(M, max_tie=0.5):
    return np.array([
        np.unique(M[:, j], return_counts=True)[1].max() / M.shape[0] <= max_tie
        for j in range(M.shape[1])
    ])


def log_norm(G):
    libs = G.sum(1, keepdims=True)
    libs[libs == 0] = 1.0
    return np.log1p(G / libs * np.median(libs))


def working_space(M, k, seed=SEED):
    Z = StandardScaler().fit_transform(gaussian_rank(M))
    Z = Z[:, np.isfinite(Z).all(0)]
    k = min(k, Z.shape[1], Z.shape[0] - 1)
    return PCA(k, random_state=seed).fit_transform(Z)


# ---------------------------------------------------------------------------
# cohort loaders -> (G_raw, X_raw, gene_symbols)
# ---------------------------------------------------------------------------
def load_kirc():
    g = ad.read_h5ad(REPO / "data/tcga_kirc/genomics/gene_expression.h5ad")
    x = ad.read_h5ad(REPO / "data/tcga_kirc/imaging/organ_radiomics.h5ad")
    pids = [p for p in g.obs_names if p in set(x.obs_names)]
    g, x = g[pids], x[pids]
    sym = np.asarray(g.var["gene_name"])
    G = log_norm(dense(g.layers["tpm_unstranded"]).astype(np.float64))
    ok = np.isfinite(G).all(0) & ((G > 0).mean(0) > 0.1)
    G, sym = G[:, ok], sym[ok]
    k = untied(G)
    return G[:, k], dense(x.X).astype(np.float64), sym[k]


def load_nsclc():
    g = ad.read_h5ad(REPO / "data/nsclc/genomics/gene_expression.h5ad")
    x = ad.read_h5ad(REPO / "data/nsclc/imaging/organ_radiomics_subset.h5ad")
    pids = [p for p in g.obs_names if p in set(x.obs_names)]
    g, x = g[pids], x[pids]
    sym = np.asarray(g.var_names)
    G = dense(g.X).astype(np.float64)
    G = np.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0)
    if G.max() > 50:  # raw counts / FPKM -> log
        G = log_norm(G)
    ok = np.isfinite(G).all(0) & ((G > 0).mean(0) > 0.1)
    G, sym = G[:, ok], sym[ok]
    k = untied(G)
    return G[:, k], dense(x.X).astype(np.float64), sym[k]


def load_adni():
    g = ad.read_h5ad(REPO / "data/adni/genomics/gene_expression.h5ad")
    x = ad.read_h5ad(REPO / "data/adni/imaging/fastsurfer_gex_pts.h5ad")
    pids = [p for p in g.obs_names if p in set(x.obs_names)]
    g, x = g[pids], x[pids]
    sym = np.asarray(g.var["symbol"]) if "symbol" in g.var else np.asarray(g.var_names)
    G = dense(g.X).astype(np.float64)
    G = np.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0)
    ok = np.isfinite(G).all(0) & (G.std(0) > 0)
    G, sym = G[:, ok], sym[ok]
    k = untied(G)
    return G[:, k], dense(x.X).astype(np.float64), sym[k]


LOADERS = {"kirc": load_kirc, "nsclc": load_nsclc, "adni": load_adni}
FIGDIR = {
    "kirc": "tcga_kirc/gene_expression/organ_radiomics",
    "nsclc": "nsclc/gene_expression/organ_radiomics_subset",
    "adni": "adni/gene_expression/fastsurfer",
}


# ---------------------------------------------------------------------------
# learning-curve experiment
# ---------------------------------------------------------------------------
def model_zoo():
    """Model families. All but boosting are fitted multi-output (all targets at once)."""
    return {
        "ridge (CV)": (lambda: RidgeCV(alphas=np.logspace(-2, 5, 40)), True),
        "kernel ridge (RBF)": (lambda: KernelRidge(
            kernel="rbf", alpha=1.0, gamma=1.0 / D_STAR), True),
        "random forest": (lambda: RandomForestRegressor(
            n_estimators=150, min_samples_leaf=5, random_state=0, n_jobs=-1), True),
        "grad. boosting": (lambda: HistGradientBoostingRegressor(
            max_iter=150, learning_rate=0.08, random_state=0), False),
    }


def _r2_cols(Y, Yhat, mu):
    """Per-column held-out R^2 against the *training* mean (no test info used)."""
    Y = np.atleast_2d(Y.T).T if Y.ndim > 1 else Y[:, None]
    Yhat = Yhat.reshape(Y.shape)
    num = np.sum((Y - Yhat) ** 2, axis=0)
    den = np.sum((Y - mu[None, :]) ** 2, axis=0)
    return 1.0 - num / den


def learning_curve(Gw, Xw, n_grid, n_targets, reps=N_REPS, seed=SEED):
    """Held-out R^2 vs training size, per model family.

    The prediction target is the leading genomic canonical direction, estimated
    on the *training* split only (nested CCA) and then applied to both splits.
    That is the direction the channel actually exposes -- an unsupervised
    genomic PC has no reason to be recoverable -- and no test patient enters its
    construction, so the protocol is leakage-free.

    Because the target direction is itself estimated, the learner pays an extra
    cost the theorem does not charge for; the observed curve is therefore a
    conservative read of the ceiling, and the fitted channel rho^2 a lower
    bound on the true one.
    """
    n = Gw.shape[0]
    zoo = model_zoo()
    out = {m: {ntr: [] for ntr in n_grid} for m in zoo}
    out["in-sample OLS"] = {ntr: [] for ntr in n_grid}
    out["CCA (held-out $\\rho^2$)"] = {ntr: [] for ntr in n_grid}
    rng = np.random.default_rng(seed)

    for ntr in n_grid:
        n_te = min(n - ntr, max(30, n // 4))
        if n_te < 15:
            continue
        K = int(min(n_targets, Gw.shape[1], Xw.shape[1]))
        for rep in range(reps):
            perm = rng.permutation(n)
            tr, te = perm[:ntr], perm[ntr:ntr + n_te]
            sc = StandardScaler().fit(Xw[tr])
            Xtr, Xte = sc.transform(Xw[tr]), sc.transform(Xw[te])

            try:
                fit = fit_recoverability(Gw[tr], Xw[tr], n_components=K)
            except Exception:
                continue
            Ytr, Yte = fit.genomic_scores(Gw[tr]), fit.genomic_scores(Gw[te])
            sd = Ytr.std(0)
            sd[sd == 0] = 1.0
            Ytr, Yte = Ytr / sd, Yte / sd
            mu = Ytr.mean(0)
            Ytr_c = Ytr - mu

            # the model-free CCA benchmark: held-out squared correlation
            Ite = fit.imaging_scores(Xw[te])
            for j in range(K):
                c = np.corrcoef(Yte[:, j], Ite[:, j])[0, 1]
                out["CCA (held-out $\\rho^2$)"][ntr].append(
                    0.0 if not np.isfinite(c) else c ** 2)

            for name, (mk, multi) in zoo.items():
                try:
                    if multi:
                        pred = mk().fit(Xtr, Ytr_c).predict(Xte) + mu
                    else:
                        pred = np.column_stack([
                            mk().fit(Xtr, Ytr_c[:, j]).predict(Xte) + mu[j]
                            for j in range(K)])
                    out[name][ntr].extend(_r2_cols(Yte, pred, mu).tolist())
                except Exception:
                    out[name][ntr].extend([np.nan] * K)

            ols = LinearRegression().fit(Xtr, Ytr_c)
            out["in-sample OLS"][ntr].extend(
                _r2_cols(Ytr, ols.predict(Xtr) + mu, mu).tolist())
    return out


def fit_ceiling(n_vals, r_vals, d_star=D_STAR):
    """One-parameter least-squares fit of R_n(rho^2) to a learning curve."""
    n_vals = np.asarray(n_vals, float)
    r_vals = np.asarray(r_vals, float)
    ok = np.isfinite(r_vals)
    n_vals, r_vals = n_vals[ok], r_vals[ok]

    def loss(logit):
        rho2 = 1.0 / (1.0 + np.exp(-logit))
        pred = np.array(
            [attainable_recoverability(rho2, n, d_star)[0] for n in n_vals])
        return float(np.sum((pred - r_vals) ** 2))

    res = minimize_scalar(loss, bounds=(-12.0, 6.0), method="bounded")
    return float(1.0 / (1.0 + np.exp(-res.x)))


# ---------------------------------------------------------------------------
def run_cohort(name):
    print(f"\n{'='*70}\n{name.upper()}\n{'='*70}")
    G_raw, X_raw, sym = LOADERS[name]()
    n = G_raw.shape[0]
    print(f"n = {n},  genes = {G_raw.shape[1]},  imaging features = {X_raw.shape[1]}")

    hv = np.argsort(G_raw.var(0))[::-1][:N_HVG]
    Gw = working_space(G_raw[:, hv], max(D_STAR, P_TARGETS + 5))
    Xw = working_space(X_raw, D_STAR)
    d_star = Xw.shape[1]
    n_targets = int(min(P_TARGETS, Gw.shape[1], d_star))
    print(f"working dims: p*={Gw.shape[1]}  d*={d_star}   "
          f"targets = top-{n_targets} training-fold canonical genomic directions")

    n_grid = sorted({int(v) for v in np.round(
        np.geomspace(max(25, d_star + 5), int(n * 0.75), 8))})
    print(f"training sizes: {n_grid}")

    curves = learning_curve(Gw, Xw, n_grid, n_targets)
    zoo_names = [k for k in curves if k != "in-sample OLS"]

    means = {m: [float(np.nanmean(curves[m][k])) if curves[m][k] else np.nan
                 for k in n_grid] for m in curves}
    ses = {m: [float(np.nanstd(curves[m][k], ddof=1)
                     / max(np.sqrt(np.sum(np.isfinite(curves[m][k]))), 1))
               if curves[m][k] else np.nan for k in n_grid] for m in curves}

    best = np.nanmax(np.vstack([means[m] for m in zoo_names]), axis=0)
    rho2_hat = fit_ceiling(n_grid, best, d_star)
    nu = float(learning_cost(rho2_hat, d_star)[0])

    # honest full-cohort spectrum for the aggregate information number
    cv = cross_validated_recoverability(Gw, Xw, n_components=min(5, d_star),
                                        n_folds=5, random_state=SEED).mean(0)
    cv = np.clip(cv, 0.0, None)

    res = {
        "cohort": name, "n": int(n), "d_star": int(d_star),
        "n_grid": n_grid,
        "mean_r2": means, "se_r2": ses,
        "best_envelope": best.tolist(),
        "rho2_channel_fit": rho2_hat,
        "nu": nu,
        "R_attainable_at_n": float(attainable_recoverability(rho2_hat, n, d_star)[0]),
        "n_for_90pct": float(sample_size_for_fraction(rho2_hat, d_star, 0.9)[0]),
        "cv_spectrum_full_cohort": cv.tolist(),
        "I_attainable_bits_at_n": attainable_information(cv, n, d_star),
        "I_channel_bits": channel_information(cv),
    }

    print(f"\n  fitted channel rho1^2 (n -> inf) = {rho2_hat:.4f}")
    print(f"  learning cost nu = {nu:.0f} patients   (half-ceiling sample size)")
    print(f"  attainable at n={n}: R = {res['R_attainable_at_n']:.4f} "
          f"({100*res['R_attainable_at_n']/max(rho2_hat,1e-12):.0f}% of channel)")
    print(f"  n needed for 90% of channel value: {res['n_for_90pct']:.0f}")
    print(f"  attainable information at n={n}: {res['I_attainable_bits_at_n']:.3f} bits"
          f"   (channel {res['I_channel_bits']:.3f} bits)")
    print(f"\n  {'n_tr':>6s} {'ceiling':>8s} " +
          " ".join(f"{m[:13]:>13s}" for m in curves))
    for i, ntr in enumerate(n_grid):
        b = attainable_recoverability(rho2_hat, ntr, d_star)[0]
        print(f"  {ntr:6d} {b:8.4f} " +
              " ".join(f"{means[m][i]:13.4f}" for m in curves))

    # ---------------- figure ----------------
    figdir = OUTROOT / FIGDIR[name]
    figdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    nd = np.geomspace(n_grid[0] * 0.8, max(n * 6, 3000), 250)
    ax.semilogx(nd, [attainable_recoverability(rho2_hat, v, d_star)[0] for v in nd],
                "k-", lw=2.4, zorder=6,
                label=rf"attainable ceiling $\mathcal{{R}}_n$ ($\hat\rho^2={rho2_hat:.3f}$)")
    ax.axhline(rho2_hat, color="k", ls=":", lw=1.5,
               label=r"fitted channel ceiling ($n=\infty$)")
    ax.plot(n_grid, means["in-sample OLS"], color="crimson", ls="--", marker="v",
            ms=4, lw=1.1, label="in-sample OLS")
    for m in zoo_names:
        ax.errorbar(n_grid, means[m], yerr=ses[m], marker="o", ms=3.5, lw=1.2,
                    capsize=2, alpha=0.9, label=m)
    ax.axvline(n, color="grey", lw=1.0, ls="-.")
    ax.text(n * 1.05, ax.get_ylim()[0] + 0.02, f"cohort $n$={n}",
            color="grey", fontsize=8)
    ax.set_xlabel("training patients $n$")
    ax.set_ylabel("held-out $R^2$ (genomic PC targets)")
    ax.set_title(f"{name.upper()}: trained models vs. the attainable ceiling")
    ax.legend(fontsize=7, loc="best", framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figdir / "attainable_bound.pdf", bbox_inches="tight")
    fig.savefig(figdir / "attainable_bound.png", dpi=160, bbox_inches="tight")
    print(f"\n  wrote {figdir/'attainable_bound.pdf'}")

    (figdir / "attainable_bound.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    which = sys.argv[1:] or ["kirc", "nsclc", "adni"]
    allres = {}
    for c in which:
        try:
            allres[c] = run_cohort(c)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"!! {c} failed: {e}")
    (OUTROOT / "attainable_bound_cohorts.json").write_text(
        json.dumps(allres, indent=2, default=float))
    print(f"\nwrote {OUTROOT/'attainable_bound_cohorts.json'}")
