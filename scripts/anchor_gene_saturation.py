"""How much of the recoverable genomic information sits in a handful of known genes?

Uses the chain rule  I(G;X) = I(G_A;X) + I(G_rest;X | G_A)  with a hand-specified
anchor panel A of canonical genes for each disease. The conditional term is
computed after residualizing both the remaining transcriptome and the imaging
features on the anchor block -- i.e. after whitening the genomics matrix against
the anchors, so that "new information" means new relative to the panel, not
merely correlated with it.

Everything is reported in *attainable* bits at the cohort's own sample size
(rgit.bounds), with a permutation null on the residual term: the question that
matters is whether anything survives beyond the panel that a cohort this size
could actually have found.

Usage:  python scripts/anchor_gene_saturation.py [nsclc kirc ...]
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

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from rgit import cross_validated_recoverability
from rgit.bounds import attainable_information, residualize
from rgit.figures import label_panels
from attainable_bound_cohorts import (  # noqa: E402  (same directory)
    LOADERS, FIGDIR, OUTROOT, gaussian_rank, working_space, N_HVG, D_STAR, SEED,
)

# ---------------------------------------------------------------------------
# hand-specified anchor panels: canonical, textbook genes for each disease
# ---------------------------------------------------------------------------
PANELS = {
    "nsclc": [
        # oncogenic drivers routinely genotyped in NSCLC
        "EGFR", "KRAS", "ALK", "MET", "BRAF", "PIK3CA", "ERBB2", "ROS1", "RET",
        # tumour suppressors
        "TP53", "STK11", "KEAP1", "RB1", "NF1", "CDKN2A", "SMARCA4",
        # histology / lineage markers a radiologist's report already proxies
        "NKX2-1", "NAPSA", "SFTPC", "TP63", "KRT5", "SOX2",
        # macroscopic programs with a plausible imaging correlate
        "MKI67", "CA9", "VEGFA", "HIF1A", "CD274", "PTPRC", "COL1A1",
    ],
    "kirc": [
        "VHL", "PBRM1", "SETD2", "BAP1", "MTOR", "TP53", "TSC1", "TSC2",
        "HIF1A", "EPAS1", "VEGFA", "CA9", "NDUFA4L2", "EGLN3", "SLC17A3",
        "MKI67", "CD274", "PTPRC", "COL1A1", "TGFB1",
    ],
    "adni": [
        "APOE", "APP", "PSEN1", "PSEN2", "MAPT", "TREM2", "CLU", "PICALM",
        "BIN1", "CR1", "ABCA7", "SORL1", "CD33", "MS4A6A", "CD2AP", "EPHA1",
        "INPP5D", "PLCG2", "SNCA", "GFAP",
    ],
}

N_PERM = 200
N_FOLDS = 5
N_PERM_SEL = 100   # permutations for the selection-aware null on the best panel
K_COMP = 3   # directions summed over: the rank theory says only a few are
             # identifiable at these n, and summing more just accumulates
             # null noise (the CV statistic is clipped at zero).


def cv_spectrum(G, X, k, n_perm=0, seed=SEED):
    """Mean K-fold held-out recoverability, optionally with a permutation null."""
    k = int(min(k, G.shape[1], X.shape[1]))
    obs = np.clip(
        cross_validated_recoverability(G, X, k, N_FOLDS, random_state=seed).mean(0),
        0.0, None)
    if not n_perm:
        return obs, None
    rng = np.random.default_rng(seed)
    null = np.zeros((n_perm, k))
    for b in range(n_perm):
        perm = rng.permutation(G.shape[0])
        null[b] = np.clip(
            cross_validated_recoverability(G[perm], X, k, N_FOLDS,
                                           random_state=seed + 1).mean(0), 0.0, None)
    return obs, null


def run(name):
    print(f"\n{'='*72}\n{name.upper()}: anchor-gene saturation\n{'='*72}")
    G_raw, X_raw, sym = LOADERS[name]()
    n = G_raw.shape[0]
    sym_u = np.array([s.upper() for s in sym.astype(str)])

    panel = PANELS[name]
    idx, found, missing = [], [], []
    for gene in panel:
        hit = np.flatnonzero(sym_u == gene.upper())
        if len(hit):
            idx.append(int(hit[0]))
            found.append(gene)
        else:
            missing.append(gene)
    print(f"n = {n};  anchor genes found: {len(found)}/{len(panel)}")
    if missing:
        print(f"  not in the expression matrix: {', '.join(missing)}")
    if len(found) < 4:
        print("  !! too few anchors, skipping")
        return None

    A_raw = G_raw[:, idx]
    rest_mask = np.ones(G_raw.shape[1], bool)
    rest_mask[idx] = False
    hv = np.argsort(G_raw[:, rest_mask].var(0))[::-1][:N_HVG]
    R_raw = G_raw[:, rest_mask][:, hv]

    # working spaces: anchors kept as themselves (rank-transformed, standardized),
    # imaging and the rest reduced to D_STAR PCs
    A = StandardScaler().fit_transform(gaussian_rank(A_raw))
    A = A[:, np.isfinite(A).all(0)]
    Xw = working_space(X_raw, D_STAR)
    d_star = Xw.shape[1]

    # --- chain rule ---------------------------------------------------------
    k_a = min(K_COMP, len(found), d_star)
    rho2_a, null_a = cv_spectrum(A, Xw, k_a, N_PERM)
    I_a = attainable_information(rho2_a, n, d_star)

    R_perp = residualize(R_raw, A)
    X_perp = residualize(Xw, A)
    Rw = working_space(R_perp, D_STAR)
    Xpw = StandardScaler().fit_transform(X_perp)
    rho2_r, null_r = cv_spectrum(Rw, Xpw, min(K_COMP, d_star), N_PERM)
    I_r = attainable_information(rho2_r, n, d_star)

    # null distribution of the residual information (is anything left at all?)
    I_r_null = np.array([attainable_information(null_r[b], n, d_star)
                         for b in range(N_PERM)])
    p_resid = (1.0 + np.sum(I_r_null >= I_r)) / (1.0 + N_PERM)
    I_a_null = np.array([attainable_information(null_a[b], n, d_star)
                         for b in range(N_PERM)])
    p_anchor = (1.0 + np.sum(I_a_null >= I_a)) / (1.0 + N_PERM)

    # Null-corrected point estimates: the permutation median is the amount of
    # apparent information this pipeline manufactures from noise at this n.
    I_a_adj = max(I_a - float(np.median(I_a_null)), 0.0)
    I_r_adj = max(I_r - float(np.median(I_r_null)), 0.0)
    total = I_a + I_r
    total_adj = I_a_adj + I_r_adj
    eta = I_a / total if total > 0 else np.nan
    eta_adj = I_a_adj / total_adj if total_adj > 0 else np.nan

    print(f"\n  working dims: |A|={A.shape[1]}  d*={d_star}")
    print(f"  I_n(anchors ; imaging)            = {I_a:.4f} bits   "
          f"(perm p = {p_anchor:.3f}, null q95 = {np.percentile(I_a_null,95):.4f})")
    print(f"  I_n(rest ; imaging | anchors)     = {I_r:.4f} bits   "
          f"(perm p = {p_resid:.3f}, null q95 = {np.percentile(I_r_null,95):.4f})")
    print(f"  total                             = {total:.4f} bits")
    print(f"  null-corrected: anchors {I_a_adj:.4f} | residual {I_r_adj:.4f} "
          f"| total {total_adj:.4f} bits")
    print(f"  anchor sufficiency  eta           = {eta:.3f} "
          f"(null-corrected {eta_adj:.3f})")

    # --- incremental saturation curve --------------------------------------
    # Rank anchors by their own single-gene held-out recoverability, then add
    # them cumulatively. Both the ranking and the choice of panel size are made
    # by looking at the data, so the honest null repeats the WHOLE procedure --
    # ranking included -- under permuted labels and takes the maximum over panel
    # size. Comparing the observed maximum against that null is what makes the
    # "best small panel" claim more than a selection artefact.
    def saturation_curve(Amat, Xmat, seed=SEED):
        singles = [float(cv_spectrum(Amat[:, [j]], Xmat, 1, seed=seed)[0][0])
                   for j in range(Amat.shape[1])]
        order = np.argsort(singles)[::-1]
        bits = []
        for m in range(1, Amat.shape[1] + 1):
            r, _ = cv_spectrum(Amat[:, order[:m]], Xmat,
                               min(m, K_COMP, d_star), seed=seed)
            bits.append(attainable_information(r, n, d_star))
        return singles, order, bits

    singles, order, cum_bits = saturation_curve(A, Xw)
    cum_genes = [found[o] for o in order]

    best_m = int(np.argmax(cum_bits)) + 1
    best_bits = float(cum_bits[best_m - 1])
    best_panel = [found[o] for o in order[:best_m]]
    rng_sel = np.random.default_rng(SEED + 7)
    sel_null = np.array([
        max(saturation_curve(A[rng_sel.permutation(n)], Xw, seed=SEED + 1)[2])
        for _ in range(N_PERM_SEL)])
    p_best = (1.0 + np.sum(sel_null >= best_bits)) / (1.0 + N_PERM_SEL)
    print(f"\n  best panel: {best_m} genes ({', '.join(best_panel)})")
    print(f"    {best_bits:.4f} bits = {100*best_bits/total:.0f}% of the cohort total; "
          f"selection-aware perm p = {p_best:.3f} "
          f"(null q95 = {np.percentile(sel_null,95):.4f})")
    print("\n  cumulative attainable bits as anchors are added "
          "(ordered by single-gene recoverability):")
    for m, (b, gname) in enumerate(zip(cum_bits, cum_genes), start=1):
        print(f"    {m:2d}  +{gname:<10s} {b:7.4f} bits"
              f"   ({100*b/total:5.1f}% of the cohort total)" if total > 0 else "")

    res = {
        "cohort": name, "n": int(n), "d_star": int(d_star),
        "panel_requested": panel, "panel_found": found, "panel_missing": missing,
        "rho2_anchor_cv": rho2_a.tolist(), "rho2_residual_cv": rho2_r.tolist(),
        "I_anchor_bits": I_a, "I_residual_bits": I_r, "I_total_bits": total,
        "eta_anchor_sufficiency": eta,
        "I_anchor_bits_null_corrected": I_a_adj,
        "I_residual_bits_null_corrected": I_r_adj,
        "I_total_bits_null_corrected": total_adj,
        "eta_null_corrected": eta_adj,
        "I_anchor_null_median": float(np.median(I_a_null)),
        "I_residual_null_median": float(np.median(I_r_null)),
        "K_components": int(K_COMP),
        "p_anchor": float(p_anchor), "p_residual": float(p_resid),
        "I_anchor_null_q95": float(np.percentile(I_a_null, 95)),
        "I_residual_null_q95": float(np.percentile(I_r_null, 95)),
        "single_gene_cv_recoverability": {found[j]: singles[j]
                                          for j in range(len(found))},
        "cumulative_bits": cum_bits,
        "best_panel_size": best_m,
        "best_panel_genes": best_panel,
        "best_panel_bits": best_bits,
        "best_panel_frac_of_total": best_bits / total if total > 0 else float("nan"),
        "best_panel_p_selection_aware": float(p_best),
        "best_panel_null_q95": float(np.percentile(sel_null, 95)),
        "cumulative_order": [found[o] for o in order],
    }

    figdir = OUTROOT / FIGDIR[name]
    figure(res, figdir)
    (figdir / "anchor_saturation.json").write_text(json.dumps(res, indent=2))
    print(f"\n  wrote {figdir/'anchor_saturation.pdf'}")
    return res


def figure(res, figdir):
    """Two panels from a cohort's saved result: cumulative saturation curve and
    the chain-rule split. Titles are left to the caption."""
    cum_bits = np.asarray(res["cumulative_bits"], dtype=float)
    total = float(res["I_total_bits"])
    best_m, best_bits = int(res["best_panel_size"]), float(res["best_panel_bits"])
    sel_null_q95 = float(res["best_panel_null_q95"])
    I_a, I_r = float(res["I_anchor_bits"]), float(res["I_residual_bits"])
    I_a_null_q95 = float(res["I_anchor_null_q95"])
    I_r_null_q95 = float(res["I_residual_null_q95"])
    figdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    ax = axes[0]
    m = np.arange(1, len(cum_bits) + 1)
    ax.plot(m, cum_bits, "o-", color="tab:blue", ms=4, label="anchor panel (cumulative)")
    ax.axhline(total, color="k", ls=":", lw=1.6,
               label=f"whole transcriptome ({total:.2f} bits)")
    ax.axhline(sel_null_q95, color="grey", ls="--", lw=1.0,
               label="selection-aware null (95%)")
    ax.plot([best_m], [best_bits], "*", color="crimson", ms=14, zorder=5,
            label=f"best panel: {best_m} genes ({100*best_bits/total:.0f}%)")
    ax.set_xlabel("number of anchor genes")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_ylim(0, max(max(cum_bits), total) * 1.6)
    ax.set_ylabel("attainable information (bits/patient)")
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.95)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.bar(["anchors\n$I(G_A;X)$", "everything else\n$I(G_{rest};X\\mid G_A)$"],
           [I_a, I_r], color=["tab:blue", "tab:orange"], alpha=0.85)
    ax.errorbar([0, 1], [I_a_null_q95, I_r_null_q95],
                fmt="_", ms=28, color="k", lw=1.6, label="permutation null (95%)")
    ax.set_ylabel("attainable information (bits/patient)")
    ax.set_ylim(0, max(I_a, I_r) * 1.3)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    label_panels(fig, axes)
    fig.savefig(figdir / "anchor_saturation.pdf", bbox_inches="tight")
    fig.savefig(figdir / "anchor_saturation.png", dpi=160, bbox_inches="tight")


if __name__ == "__main__":
    if "--replot" in sys.argv:      # redraw from saved JSON, no recomputation
        saved = json.load(open(OUTROOT / "anchor_saturation.json"))
        for c in [a for a in sys.argv[1:] if not a.startswith("--")] or list(saved):
            figure(saved[c], OUTROOT / FIGDIR[c])
        sys.exit(0)
    which = sys.argv[1:] or ["nsclc", "kirc"]
    out = {}
    for c in which:
        try:
            r = run(c)
            if r:
                out[c] = r
        except Exception:
            import traceback
            traceback.print_exc()
    (OUTROOT / "anchor_saturation.json").write_text(
        json.dumps(out, indent=2, default=float))
    print(f"\nwrote {OUTROOT/'anchor_saturation.json'}")
