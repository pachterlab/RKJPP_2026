"""Per-gene classification sweep: how well does imaging predict each gene?

The stress test (scripts/bound_stress_test.py) scores learners on canonical
directions in bits. This script asks the question the radiogenomics literature
usually asks, gene by gene: for each of the 2000 highly variable genes, binarize
expression at its median and classify it from the d* imaging components with
L2 logistic regression under stratified 5-fold cross-validation. The result is
the empirical distribution of per-gene AUCs that a supervised radiogenomic
study would report, on every cohort, next to

  * a patient-label permutation null (the imaging rows are shuffled, the whole
    2000-gene sweep is repeated), which gives the AUC distribution and the
    best-of-2000 AUC expected when there is no signal at all, and
  * the AUC ceiling of the manuscript, 1/2 + (2/pi) arcsin(sqrt(R_n/2)), at the
    learning-curve fit and at the model-free 95% upper limit on the leading
    channel value (read from notebooks/figures/attainable_summary.json).

Where covariates are available (TCGA-KIRC: age, sex, ethnicity, race, scanner
manufacturer; ADNI: age, sex, education) the sweep is repeated after
residualizing the covariates from both modalities, so that sex-linked genes,
which brain morphometry and whole-body CT read out directly, are separated
from the rest.

Usage:  python scripts/per_gene_auc_sweep.py [cohort ...]      (default: all)
        python scripts/per_gene_auc_sweep.py --replot          (redraw from JSON)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from rgit.figures import label_panels
from attainable_bound_cohorts import (  # noqa: E402
    LOADERS, OUTROOT, REPO, working_space, D_STAR, SEED,
)

N_HVG = 2000
N_FOLDS = 5
N_PERM = 10          # whole-sweep permutations of the patient labels
COHORT_LABEL = {"kirc": "TCGA-KIRC (CT)", "nsclc": "NSCLC (CT)", "adni": "ADNI (MRI)"}


# ---------------------------------------------------------------------------
# covariates
# ---------------------------------------------------------------------------
def patient_ids(cohort):
    paths = {
        "kirc": ("data/tcga_kirc/genomics/gene_expression.h5ad",
                 "data/tcga_kirc/imaging/organ_radiomics.h5ad"),
        "nsclc": ("data/nsclc/genomics/gene_expression.h5ad",
                  "data/nsclc/imaging/organ_radiomics_subset.h5ad"),
        "adni": ("data/adni/genomics/gene_expression.h5ad",
                 "data/adni/imaging/fastsurfer_gex_pts.h5ad"),
    }[cohort]
    g = ad.read_h5ad(REPO / paths[0], backed="r")
    x = ad.read_h5ad(REPO / paths[1], backed="r")
    xs = set(x.obs_names)
    pids = [p for p in g.obs_names if p in xs]
    ycoll = None
    if cohort == "adni" and "YearofCollection" in g.obs:
        ycoll = pd.to_numeric(g.obs.loc[pids, "YearofCollection"], errors="coerce")
    return pids, ycoll


def covariates(cohort, pids, ycoll=None):
    """Design matrix [1, covariates] aligned to pids, or None if unavailable."""
    if cohort == "kirc":
        from kirc_deconfound import confounders
        D, names = confounders(pids)
        return D, names
    if cohort == "adni":
        demo = (pd.read_csv(REPO / "data/adni/demographics.csv", low_memory=False)
                .drop_duplicates("PTID").set_index("PTID"))
        yob = pd.to_datetime(demo["PTDOBYY"], errors="coerce").dt.year
        yob = yob.fillna(pd.to_numeric(demo["PTDOBYY"], errors="coerce"))
        yc = ycoll.fillna(2011.0).values if ycoll is not None else np.full(len(pids), 2011.0)
        C = pd.DataFrame({
            "age": yc - yob.reindex(pids).values,
            "sex": pd.to_numeric(demo["PTGENDER"], errors="coerce").reindex(pids).values,
            "education": pd.to_numeric(demo["PTEDUCAT"], errors="coerce")
                           .replace(-4, np.nan).reindex(pids).values,
        }, index=pids)
        C = C.fillna(C.mean())
        Z = StandardScaler().fit_transform(C.values)
        return np.column_stack([np.ones(len(pids)), Z]), list(C.columns)
    if cohort == "nsclc":
        # The cohort was released in two tranches (98 original patients, 31
        # later cases); patient numbers R01-128 and above are the later tranche.
        late = np.array([int(p.split("-")[1]) >= 128 for p in pids], float)
        return np.column_stack([np.ones(len(pids)), late]), ["release tranche"]
    return None, []


def residualize(M, D):
    beta, *_ = np.linalg.lstsq(D, M, rcond=None)
    return M - D @ beta


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------
def _fold_aucs(Xtr, Xte, Ytr, Yte, seed):
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    out = np.full(Ytr.shape[1], np.nan)
    for j in range(Ytr.shape[1]):
        ytr, yte = Ytr[:, j], Yte[:, j]
        if ytr.min() == ytr.max() or yte.min() == yte.max():
            continue
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
        clf.fit(Xtr, ytr)
        out[j] = roc_auc_score(yte, clf.decision_function(Xte))
    return out


def sweep(Xw, Y, seed=SEED, n_jobs=-1):
    """Mean 5-fold AUC for every column of the binary matrix Y."""
    # Folds are stratified on the first gene; every gene is median-binarized so
    # the class balance is ~1/2 for all of them and one split serves all.
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    splits = list(skf.split(Xw, Y[:, 0]))
    chunks = np.array_split(np.arange(Y.shape[1]), max(1, 8 * N_FOLDS))
    jobs = [(tr, te, ch) for tr, te in splits for ch in chunks]
    res = Parallel(n_jobs=n_jobs)(
        delayed(_fold_aucs)(Xw[tr], Xw[te], Y[tr][:, ch], Y[te][:, ch], seed)
        for tr, te, ch in jobs)
    acc = np.zeros((N_FOLDS, Y.shape[1]))
    k = 0
    for f in range(N_FOLDS):
        for ch in chunks:
            acc[f, ch] = res[k]; k += 1
    return np.nanmean(acc, axis=0)


def binarize(G):
    return (G > np.median(G, axis=0, keepdims=True)).astype(int)


def run(cohort):
    print(f"\n{'='*78}\n{cohort.upper()}: per-gene AUC sweep\n{'='*78}")
    G_raw, X_raw, sym = LOADERS[cohort]()
    n = G_raw.shape[0]
    order = np.argsort(G_raw.var(0))[::-1][:N_HVG]
    G, sym = G_raw[:, order], np.asarray(sym)[order].astype(str)
    Xw = working_space(X_raw, D_STAR)
    print(f"  n={n}, d*={Xw.shape[1]}, genes={G.shape[1]}")

    rng = np.random.default_rng(SEED)
    Y = binarize(G)
    auc = sweep(Xw, Y)
    print(f"  observed: median {np.nanmedian(auc):.3f}, "
          f"95th pct {np.nanpercentile(auc, 95):.3f}, max {np.nanmax(auc):.3f}")

    null = []
    for b in range(N_PERM):
        null.append(sweep(Xw[rng.permutation(n)], Y, seed=SEED + 1 + b))
        print(f"  permutation {b+1}/{N_PERM}: max {np.nanmax(null[-1]):.3f}")
    null = np.asarray(null)
    null_q95 = float(np.nanpercentile(null, 95))
    null_max = np.nanmax(null, axis=1)

    out = {
        "cohort": cohort, "n": int(n), "d_star": int(Xw.shape[1]),
        "n_genes": int(G.shape[1]), "n_perm": N_PERM, "n_folds": N_FOLDS,
        "genes": sym.tolist(), "auc": auc.tolist(),
        "null_auc_q50": float(np.nanmedian(null)), "null_auc_q95": null_q95,
        "null_auc_q99": float(np.nanpercentile(null, 99)),
        "null_max_auc": null_max.tolist(),
        "null_max_auc_q95": float(np.percentile(null_max, 95)),
        "null_hist": np.histogram(null[np.isfinite(null)], bins=np.linspace(0.3, 0.9, 61))[0].tolist(),
        "summary": summarize(auc, sym, null_q95, null_max),
    }

    out["shared_axis"] = shared_axis(G, Xw, Y)
    print(f"  leading expression axis: {out['shared_axis']}")

    D, names = covariates(cohort, *patient_ids(cohort))
    if D is not None:
        Ga, Xa = residualize(G, D), residualize(Xw, D)
        auc_adj = sweep(Xa, binarize(Ga))
        out["adjusted_for"] = names
        out["auc_adjusted"] = auc_adj.tolist()
        out["summary_adjusted"] = summarize(auc_adj, sym, null_q95, null_max)
        print(f"  adjusted for {', '.join(names)}: median {np.nanmedian(auc_adj):.3f}, "
              f"max {np.nanmax(auc_adj):.3f}")
    return out


def shared_axis(G, Xw, Y):
    """How much of the per-gene sweep is one test repeated.

    The leading principal axis of expression (often a sequencing-depth or
    complexity axis) can dominate every median-binarized label. Reports its
    variance share, the agreement of each gene's label with the axis label
    (|phi| correlation, median over genes) and the cross-validated AUC with
    which imaging classifies the axis label itself.
    """
    from sklearn.decomposition import PCA
    Z = StandardScaler().fit_transform(G)
    p = PCA(1).fit(Z)
    s = p.transform(Z)[:, 0]
    y1 = (s > np.median(s)).astype(int)
    yc = 2 * Y - 1
    phi = np.abs(((2 * y1 - 1)[:, None] * yc).mean(0))
    auc1 = sweep(Xw, y1[:, None])[0]
    return {"pc1_variance_ratio": float(p.explained_variance_ratio_[0]),
            "corr_pc1_total_log_expression": float(np.corrcoef(s, G.sum(1))[0, 1]),
            "median_abs_phi_gene_vs_pc1": float(np.median(phi)),
            "frac_genes_phi_ge_0.5": float(np.mean(phi >= 0.5)),
            "auc_pc1_label": float(auc1)}


def summarize(auc, sym, null_q95, null_max, k=10):
    ok = np.isfinite(auc)
    a, s = auc[ok], sym[ok]
    top = np.argsort(a)[::-1][:k]
    return {
        "median": float(np.median(a)), "mean": float(a.mean()),
        "q95": float(np.percentile(a, 95)), "max": float(a.max()),
        "frac_ge_0.6": float(np.mean(a >= 0.6)), "frac_ge_0.7": float(np.mean(a >= 0.7)),
        "n_above_null_q95": int(np.sum(a > null_q95)),
        "expected_above_null_q95": float(0.05 * len(a)),
        "n_above_null_max_q95": int(np.sum(a > np.percentile(null_max, 95))),
        "top_genes": [{"gene": str(s[i]), "auc": float(a[i])} for i in top],
    }


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def figure(all_res):
    ceilings = {r["cohort"]: r for r in json.load(open(OUTROOT / "attainable_summary.json"))}
    cohorts = [c for c in ("kirc", "nsclc", "adni") if c in all_res]
    bins = np.linspace(0.3, 0.9, 61)
    fig, axes = plt.subplots(1, len(cohorts), figsize=(5.0 * len(cohorts), 3.9),
                             squeeze=False, sharey=True)
    for ax, c in zip(axes[0], cohorts):
        r = all_res[c]
        auc = np.asarray(r["auc"], float)
        auc = auc[np.isfinite(auc)]
        null = np.asarray(r["null_hist"], float) / r["n_perm"]
        ax.bar(0.5 * (bins[1:] + bins[:-1]), null, width=np.diff(bins), color="0.72",
               label="permutation null", zorder=1)
        ax.hist(auc, bins=bins, color="tab:blue", alpha=0.85,
                label="observed, 2000 HVGs", zorder=2)
        if "auc_adjusted" in r:
            adj = np.asarray(r["auc_adjusted"], float)
            ax.hist(adj[np.isfinite(adj)], bins=bins, histtype="step", lw=1.4,
                    color="tab:orange", zorder=3, label="covariate-adjusted")
        ax.axvline(0.5, color="0.3", ls=":", lw=1.0, zorder=4)
        ax.axvline(r["null_max_auc_q95"], color="0.3", ls="--", lw=1.0, zorder=4,
                   label="best of 2000 under the null (95th pct)")
        ce = ceilings[c]
        ax.axvline(ce["auc_ceiling_ucl"], color="k", lw=2.2, zorder=5,
                   label=r"AUC ceiling at $\bar\rho_1^2$ (95% UCL)")
        ax.axvline(ce["auc_ceiling_fit"], color="k", lw=1.2, ls="-.", zorder=5,
                   label=r"AUC ceiling at $\hat\rho_1^2$ (fit)")
        s = r["summary"]
        t = s["top_genes"][0]
        ax.annotate(t["gene"], xy=(t["auc"], 0), xytext=(-6, 22), textcoords="offset points",
                    ha="right", fontsize=7, style="italic",
                    arrowprops=dict(arrowstyle="-", color="0.4", lw=0.6))
        sa = r["shared_axis"]
        ax.plot([sa["auc_pc1_label"]], [0], marker="^", color="tab:red", ms=7,
                clip_on=False, zorder=6, ls="none",
                label="leading expression axis, binarized")
        lines = [f"{COHORT_LABEL[c]}, $n$ = {r['n']}",
                 f"median AUC {s['median']:.2f}, max {s['max']:.2f}",
                 f"{s['n_above_null_q95']} of {r['n_genes']} genes above null 95th pct "
                 f"({s['expected_above_null_q95']:.0f} expected)"]
        if "summary_adjusted" in r:
            sa2 = r["summary_adjusted"]
            lines.append(f"adjusted for {', '.join(r['adjusted_for'])}: "
                         f"median {sa2['median']:.2f}, max {sa2['max']:.2f}")
        ax.text(0.03, 0.96, "\n".join(lines), transform=ax.transAxes, fontsize=7.2,
                va="top", bbox=dict(fc="white", ec="0.8", alpha=0.9, pad=3))
        ax.set_xlim(0.3, 0.9)
        ax.set_xlabel("cross-validated AUC, median-binarized expression")
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel("genes")
    h, l = axes[0][-1].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    label_panels(fig, axes[0])
    fig.savefig(OUTROOT / "per_gene_auc_sweep.pdf", bbox_inches="tight")
    fig.savefig(OUTROOT / "per_gene_auc_sweep.png", dpi=160, bbox_inches="tight")
    print(f"\nwrote {OUTROOT/'per_gene_auc_sweep.pdf'}")


if __name__ == "__main__":
    path = OUTROOT / "per_gene_auc_sweep.json"
    if "--replot" in sys.argv:
        figure(json.load(open(path)))
        sys.exit(0)
    prev = json.load(open(path)) if path.exists() else {}
    for c in (sys.argv[1:] or ["kirc", "nsclc", "adni"]):
        prev[c] = run(c)
        path.write_text(json.dumps(prev, indent=1, default=float))
    figure(prev)
