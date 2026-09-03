"""Target-class analysis on TCGA-KIRC: can CT decode the *clinically actionable*
ccRCC targets, as opposed to transcriptome variance in aggregate?

Three target classes are decoded from the same four CT representations used in the
transcriptome-wide analysis:

  1. somatic driver mutation status (VHL, PBRM1, BAP1, SETD2) -- the alterations that
     ccRCC radiogenomic "virtual biopsy" studies most often claim to predict;
  2. treatment-selection expression signatures (IMmotion151 angiogenesis, T-effector,
     myeloid inflammation) -- the scores that stratify tyrosine-kinase-inhibitor versus
     immune-checkpoint benefit;
  3. macroscopic positive controls (pathologic stage III-IV, patient sex, age) --
     targets CT is expected to carry, which calibrate whether the pipeline can detect
     signal at all.

Every target uses the identical estimator, cross-validation split, and permutation null,
so the resulting numbers are directly comparable across target classes.

    python scripts/kirc_target_classes.py
"""

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.special import ndtri
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "notebooks/figures"

SEED = 0
TAU = 5           # patients per working dimension
N_PERM = 200      # permutations per null
N_HVG = 2000

IMAGING = ["tumor_radiomics", "organ_radiomics", "tumor_radimagenet", "whole_radimagenet"]
PRETTY = {
    "tumor_radiomics": "Tumor radiomics",
    "organ_radiomics": "Kidney radiomics",
    "tumor_radimagenet": "Tumor deep",
    "whole_radimagenet": "Whole-volume deep",
}

# IMmotion150/151 treatment-selection signatures (Motzer et al., Nat Med 2020).
SIGNATURES = {
    "angiogenesis": ["VEGFA", "KDR", "ESM1", "PECAM1", "ANGPTL4", "CD34"],
    "t_effector": ["CD8A", "EOMES", "PRF1", "IFNG", "CD274"],
    "myeloid_inflammation": ["IL6", "CXCL1", "CXCL2", "CXCL3", "CXCL8", "PTGS2"],
}
DRIVERS = ["VHL", "PBRM1", "BAP1", "SETD2"]

ALPHAS = np.logspace(-1, 5, 25)
CS = np.logspace(-4, 2, 13)

dense = lambda M: M.toarray() if sp.issparse(M) else np.asarray(M)


def gauss_rank(M):
    """Rank-based inverse-normal transform, column-wise."""
    M = np.asarray(M, float)
    n = M.shape[0]
    return ndtri(np.apply_along_axis(lambda c: rankdata(c, method="average"), 0, M) / (n + 1.0))


def untied(M, max_tie=0.5):
    """Drop columns whose modal value covers more than `max_tie` of patients."""
    return np.array([np.unique(M[:, j], return_counts=True)[1].max() / M.shape[0] <= max_tie
                     for j in range(M.shape[1])])


def working(M, k, seed=SEED):
    """Gaussian-rank, standardize, then PCA to k components."""
    Z = StandardScaler().fit_transform(gauss_rank(M))
    Z = Z[:, np.isfinite(Z).all(0)]
    return PCA(min(k, Z.shape[1], Z.shape[0] - 1), random_state=seed).fit_transform(Z)


def resid(M, D):
    return M - D @ np.linalg.lstsq(D, M, rcond=None)[0]


# --------------------------------------------------------------------------- scorers
def cv_r2(X, y, cv):
    """Out-of-fold R^2, ridge penalty chosen by out-of-fold R^2 (same grid every time)."""
    best = (None, -np.inf)
    for al in ALPHAS:
        p = np.zeros_like(y, dtype=float)
        for tr, te in cv.split(X, y):
            p[te] = Ridge(alpha=al).fit(X[tr], y[tr]).predict(X[te])
        r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        if r2 > best[1]:
            best = (al, r2)
    return best[1]


def cv_auc(X, y, cv):
    """Out-of-fold AUC, L2 penalty chosen by out-of-fold AUC."""
    best = -np.inf
    for c in CS:
        p = np.zeros(len(y))
        for tr, te in cv.split(X, y):
            clf = LogisticRegression(C=c, max_iter=5000).fit(X[tr], y[tr])
            p[te] = clf.predict_proba(X[te])[:, 1]
        best = max(best, roc_auc_score(y, p))
    return best


def permutation_null(score_fn, X, y, cv, n_perm=N_PERM, seed=SEED):
    """Null distribution from permuting patients in the image matrix."""
    rng = np.random.default_rng(seed)
    obs = score_fn(X, y, cv)
    null = np.array([score_fn(X[rng.permutation(len(y))], y, cv) for _ in range(n_perm)])
    p = (1 + (null >= obs).sum()) / (1 + n_perm)
    return dict(observed=float(obs), null_mean=float(null.mean()),
                null_q95=float(np.quantile(null, 0.95)), p=float(p))


# --------------------------------------------------------------------------- data
def main():
    OUT.mkdir(exist_ok=True)

    gen = ad.read_h5ad(REPO / "data/tcga_kirc/genomics/gene_expression.h5ad")
    symbols_all = np.asarray(gen.var["gene_name"])
    G_tpm = dense(gen.layers["tpm_unstranded"]).astype(np.float64)
    libs = G_tpm.sum(1, keepdims=True)
    libs[libs == 0] = 1.0
    G_log = np.log1p(G_tpm / libs * np.median(libs))
    expressed = np.isfinite(G_log).all(0) & ((G_log > 0).mean(0) > 0.10)
    G_log, symbols = G_log[:, expressed], symbols_all[expressed]

    patients = list(gen.obs_names)
    n = len(patients)
    p_star = n // TAU
    print(f"n = {n} patients, working dimension = {p_star}")

    # ---- CT representations, reduced exactly as in the transcriptome-wide analysis
    Xw = {}
    for name in IMAGING:
        a = ad.read_h5ad(REPO / f"data/tcga_kirc/imaging/{name}.h5ad")[patients]
        M = dense(a.X).astype(np.float64)
        M = M[:, M.std(0) > 0]
        M = M[:, untied(M)]
        Xw[name] = working(M, p_star)
        print(f"  {name:22s} -> {Xw[name].shape[1]} components")

    # ---- clinical covariates
    cols = ["cases.submitter_id", "demographic.age_at_index", "demographic.gender",
            "demographic.ethnicity", "demographic.race", "diagnoses.ajcc_pathologic_stage"]
    clin = (pd.read_csv(REPO / "data/tcga_kirc/clinical_tcga.tsv", sep="\t",
                        usecols=cols, low_memory=False)
            .replace("'--", np.nan).drop_duplicates("cases.submitter_id")
            .set_index("cases.submitter_id").reindex(patients))

    digest = pd.read_excel(REPO / "data/tcga_kirc/imaging/TCIA_TCGA-KIRC_09-16-2015-nbia-digest.xlsx")
    digest["Manufacturer"] = digest["Manufacturer"].str.replace(
        "Philips Medical Systems", "Philips", regex=False)
    manufacturer = (digest.groupby("Patient ID")["Manufacturer"]
                    .agg(lambda s: s.dropna().mode().iloc[0] if len(s.dropna()) else np.nan)
                    .reindex(patients))

    age = pd.to_numeric(clin["demographic.age_at_index"], errors="coerce")
    age = age.fillna(age.median()).values
    sex_female = (clin["demographic.gender"] == "female").astype(int).values
    stage = clin["diagnoses.ajcc_pathologic_stage"].fillna("unknown").astype(str)
    stage_known = (stage != "unknown").values
    advanced = stage.str.contains("III|IV", regex=True).astype(int).values

    cat = pd.DataFrame({
        "sex": clin["demographic.gender"].fillna("unknown"),
        "eth": clin["demographic.ethnicity"].fillna("not reported"),
        "race": clin["demographic.race"].fillna("not reported"),
        "manu": manufacturer.fillna("unknown"),
    }, index=patients)
    age_z = StandardScaler().fit_transform(age.reshape(-1, 1))
    D_DEMO = np.column_stack([np.ones(n), age_z,
                              pd.get_dummies(cat[["sex", "eth", "race"]], drop_first=True)
                              .values.astype(float)])
    D_ALL = np.column_stack([D_DEMO,
                             pd.get_dummies(cat[["manu"]], drop_first=True).values.astype(float)])

    results = {"cohort": {"n": int(n), "p_star": int(p_star)}, "targets": {}}

    # ---- 1. expression signatures (all 190 patients) ------------------------
    sym_idx = {s: j for j, s in enumerate(symbols)}
    Gz = StandardScaler().fit_transform(gauss_rank(G_log))
    kf = KFold(5, shuffle=True, random_state=SEED)

    for sig, genes in SIGNATURES.items():
        idx = [sym_idx[g] for g in genes if g in sym_idx]
        y = Gz[:, idx].mean(1)
        y = (y - y.mean()) / y.std()
        entry = {"kind": "signature", "n": int(n), "n_genes": len(idx),
                 "genes": [g for g in genes if g in sym_idx], "by_image": {}}
        for name in IMAGING:
            entry["by_image"][name] = permutation_null(cv_r2, Xw[name], y, kf)
        best = max(entry["by_image"], key=lambda k: entry["by_image"][k]["observed"])
        entry["best_image"] = best
        # demographic + scanner adjustment on the strongest representation
        entry["deconf_demo_r2"] = float(cv_r2(resid(Xw[best], D_DEMO), resid(y, D_DEMO), kf))
        entry["deconf_all_r2"] = float(cv_r2(resid(Xw[best], D_ALL), resid(y, D_ALL), kf))
        results["targets"][sig] = entry
        b = entry["by_image"][best]
        print(f"[signature {sig:22s}] best={PRETTY[best]:18s} CV R2={b['observed']:+.3f} "
              f"(null {b['null_mean']:+.3f}, p={b['p']:.3f}) "
              f"-demo {entry['deconf_demo_r2']:+.3f} -demo+scanner {entry['deconf_all_r2']:+.3f}")

    # ---- 2. somatic driver mutation status ---------------------------------
    mut = ad.read_h5ad(REPO / "data/tcga_kirc/genomics/mutated_genes.h5ad")
    shared = [p for p in patients if p in set(mut.obs_names)]
    rows = np.array([patients.index(p) for p in shared])
    Mm = dense(mut[shared].X)
    mut_genes = list(mut.var_names)
    n_mut_cohort = len(shared)
    p_star_mut = n_mut_cohort // TAU
    results["cohort"]["n_mutation"] = int(n_mut_cohort)
    results["cohort"]["p_star_mutation"] = int(p_star_mut)
    print(f"\nmutation subcohort: n = {n_mut_cohort}, working dimension = {p_star_mut}")

    Xm = {name: Xw[name][rows][:, :p_star_mut] for name in IMAGING}
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)

    for g in DRIVERS:
        if g not in mut_genes:
            continue
        y = (Mm[:, mut_genes.index(g)] > 0).astype(int)
        entry = {"kind": "mutation", "n": int(n_mut_cohort), "n_mutated": int(y.sum()),
                 "prevalence": float(y.mean()), "by_image": {}}
        for name in IMAGING:
            entry["by_image"][name] = permutation_null(cv_auc, Xm[name], y, skf)
        best = max(entry["by_image"], key=lambda k: entry["by_image"][k]["observed"])
        entry["best_image"] = best
        results["targets"][f"mut_{g}"] = entry
        b = entry["by_image"][best]
        print(f"[mutation  {g:22s}] best={PRETTY[best]:18s} CV AUC={b['observed']:.3f} "
              f"(null {b['null_mean']:.3f}, q95 {b['null_q95']:.3f}, p={b['p']:.3f}) "
              f"[{entry['n_mutated']}/{entry['n']} mutated]")

    # ---- 3. macroscopic positive controls -----------------------------------
    controls = {
        "stage_III_IV": ("binary", advanced[stage_known], stage_known),
        "sex_female": ("binary", sex_female, np.ones(n, bool)),
        "age": ("continuous", age, np.ones(n, bool)),
    }
    for name_t, (kind, y, mask) in controls.items():
        entry = {"kind": f"control_{kind}", "n": int(mask.sum()), "by_image": {}}
        for name in IMAGING:
            Xc = Xw[name][mask]
            if kind == "binary":
                entry["by_image"][name] = permutation_null(cv_auc, Xc, np.asarray(y), skf)
            else:
                yy = np.asarray(y, float)
                yy = (yy - yy.mean()) / yy.std()
                entry["by_image"][name] = permutation_null(cv_r2, Xc, yy, kf)
        best = max(entry["by_image"], key=lambda k: entry["by_image"][k]["observed"])
        entry["best_image"] = best
        results["targets"][name_t] = entry
        b = entry["by_image"][best]
        metric = "AUC" if kind == "binary" else "R2"
        print(f"[control   {name_t:22s}] best={PRETTY[best]:18s} CV {metric}={b['observed']:+.3f} "
              f"(null {b['null_mean']:+.3f}, p={b['p']:.3f})")

    with open(OUT / "kirc_target_classes.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT / 'actionable_targets.json'}")


if __name__ == "__main__":
    main()
