"""Cross-cohort summary: the headline attainable-information numbers.

Collects the outputs of attainable_bound_cohorts.py (learning-curve fit) and
channel_ucl.py (model-free upper confidence limit) into one table and one
figure. The two bracket the channel:

  rho^2_fit   -- the smallest channel value consistent with what trained models
                 actually achieved (a lower read),
  rho^2_UCL   -- the largest consistent with the data at 95% (an upper read),

and the reportable ceiling is R_n / I_n evaluated at the UCL.

Also converts the bits into the unit a clinician can act on: since a perfectly
recovered 50/50 binary call carries exactly one bit, I_n *is* the number of
ideal binary biomarkers the imaging study is worth.
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

from rgit.figures import label_panels
from rgit.bounds import (
    attainable_recoverability, attainable_information, auc_ceiling,
    channel_information, learning_cost, sample_size_for_fraction,
)

OUT = Path("notebooks/figures")
COHORTS = ["kirc", "nsclc", "adni"]
LABEL = {"kirc": "TCGA-KIRC (CT)", "nsclc": "NSCLC (CT)", "adni": "ADNI (MRI)"}


def main():
    ucl = json.load(open(OUT / "channel_ucl.json"))
    coh = json.load(open(OUT / "attainable_bound_cohorts.json"))
    anc = json.load(open(OUT / "anchor_saturation.json")) if (
        OUT / "anchor_saturation.json").exists() else {}
    tot = json.load(open(OUT / "total_information_ucl.json")) if (
        OUT / "total_information_ucl.json").exists() else {}

    rows = []
    for c in COHORTS:
        u, k = ucl[c], coh[c]
        n, d = u["n"], u["d_star"]
        lo, hi = k["rho2_channel_fit"], u["channel_rho2_ucl95"]
        Rn_hi = float(attainable_recoverability(hi, n, d)[0])
        Rn_lo = float(attainable_recoverability(lo, n, d)[0])
        rows.append({
            "cohort": c, "label": LABEL[c], "n": n, "d_star": d,
            "rho2_fit": lo, "rho2_ucl95": hi,
            "R_attainable_fit": Rn_lo, "R_attainable_ucl": Rn_hi,
            "I_attainable_bits_fit": attainable_information([lo], n, d),
            "I_attainable_bits_ucl": attainable_information([hi], n, d),
            "I_channel_bits_ucl": channel_information([hi]),
            "auc_ceiling_fit": float(auc_ceiling([Rn_lo])[0]),
            "auc_ceiling_ucl": float(auc_ceiling([Rn_hi])[0]),
            "nu_fit": float(learning_cost(lo, d)[0]),
            "n_for_90pct_fit": float(sample_size_for_fraction(lo, d, 0.9)[0]),
            "observed_R1_cv": u["observed_R1_cv"],
            "perm_null_q95": u["perm_null_q95"],
            "theory_dominates_fraction": u["theory_dominates_fraction"],
            # total-retention limit: the number to quote (leading-axis I_n
            # bounds only the best canonical direction, so it is a LOWER bound
            # on the total)
            "I_total_ucl_bits": tot.get(c, {}).get("total_information_ucl_bits"),
            "I_total_observed_bits": tot.get(c, {}).get("observed_plugin_bits"),
            "I_total_null_median_bits": tot.get(c, {}).get("null_median_bits"),
        })
        if c in anc:
            a = anc[c]
            rows[-1].update({
                "anchor_best_panel_size": a.get("best_panel_size"),
                "anchor_best_panel_genes": a.get("best_panel_genes"),
                "anchor_best_frac_of_total": a.get("best_panel_frac_of_total"),
                "anchor_best_p": a.get("best_panel_p_selection_aware"),
                "eta_anchor": a.get("eta_anchor_sufficiency"),
            })

    hdr = (f"{'cohort':16s} {'n':>5s} {'rho2 fit':>9s} {'rho2 UCL':>9s} "
           f"{'R_n UCL':>8s} {'I_lead':>8s} {'I_TOTAL':>8s} {'AUC ceil':>9s} "
           f"{'nu':>6s} {'n(90%)':>8s}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['label']:16s} {r['n']:5d} {r['rho2_fit']:9.3f} "
              f"{r['rho2_ucl95']:9.3f} {r['R_attainable_ucl']:8.3f} "
              f"{r['I_attainable_bits_ucl']:8.3f} "
              f"{(r['I_total_ucl_bits'] if r['I_total_ucl_bits'] is not None else float('nan')):8.3f} "
              f"{r['auc_ceiling_ucl']:9.3f} "
              f"{r['nu_fit']:6.0f} {r['n_for_90pct_fit']:8.0f}")

    print("\nHeadline (95% upper confidence limits on TOTAL retention):")
    for r in rows:
        tb = r["I_total_ucl_bits"]
        flag = ""
        if (r["I_total_observed_bits"] is not None
                and r["I_total_null_median_bits"] is not None
                and r["I_total_observed_bits"] <= r["I_total_null_median_bits"]):
            flag = "  [observed statistic BELOW its permutation null]"
        print(f"  {r['label']:16s} n={r['n']:4d}:  <= {tb:.2f} bits/patient "
              f"(<= {tb:.2f} ideal binary biomarkers);  any single-feature "
              f"classifier AUC <= {r['auc_ceiling_ucl']:.2f}{flag}")

    # ---------------- figure ----------------
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))

    ax = axes[0]
    nd = np.geomspace(20, 20000, 300)
    colors = {"kirc": "tab:blue", "nsclc": "tab:orange", "adni": "tab:green"}
    for r in rows:
        c = r["cohort"]
        y = [attainable_information([r["rho2_ucl95"]], v, r["d_star"]) for v in nd]
        ax.semilogx(nd, y, color=colors[c], lw=2.0, label=f"{r['label']} (95% UCL)")
        y2 = [attainable_information([r["rho2_fit"]], v, r["d_star"]) for v in nd]
        ax.semilogx(nd, y2, color=colors[c], lw=1.2, ls="--", alpha=0.75)
        ax.plot([r["n"]], [r["I_attainable_bits_ucl"]], "o", color=colors[c], ms=7,
                markeredgecolor="k", zorder=5)
        ax.annotate(f"{r['I_attainable_bits_ucl']:.2f} bits",
                    (r["n"], r["I_attainable_bits_ucl"]),
                    textcoords="offset points", xytext=(6, -11), fontsize=7.5,
                    color=colors[c])
    ax.set_xlabel("cohort size $n$")
    ax.set_ylabel(r"attainable information $\mathcal{I}_n$ (bits/patient)")
    ax.legend(fontsize=7.2, loc="upper left")
    ax.grid(alpha=0.25)
    ax.text(0.98, 0.04, "solid: 95% UCL   dashed: learning-curve fit",
            transform=ax.transAxes, ha="right", fontsize=7, color="grey")

    ax = axes[1]
    x = np.arange(len(rows))
    ax.bar(x - 0.19, [r["R_attainable_ucl"] for r in rows], 0.36,
           label=r"attainable ceiling $\mathcal{R}_n$ (95% UCL)",
           color="lightsteelblue", edgecolor="k", lw=0.6)
    ax.bar(x + 0.19, [r["observed_R1_cv"] for r in rows], 0.36,
           label=r"observed $\hat R_1^{\mathrm{cv}}$", color="tab:blue")
    ax.errorbar(x + 0.19, [r["perm_null_q95"] for r in rows], fmt="_", ms=20,
                color="crimson", lw=1.6, label="permutation null (95%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['label']}\n$n$={r['n']}" for r in rows], fontsize=8)
    ax.set_ylabel("leading recoverability")
    ax.set_ylim(0, max(r["R_attainable_ucl"] for r in rows) * 1.42)
    ax.legend(fontsize=7.2, loc="upper left", ncol=1, framealpha=0.95)
    ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    label_panels(fig, axes)
    fig.savefig(OUT / "attainable_summary.pdf", bbox_inches="tight")
    fig.savefig(OUT / "attainable_summary.png", dpi=160, bbox_inches="tight")
    (OUT / "attainable_summary.json").write_text(json.dumps(rows, indent=2, default=float))

    # LaTeX table for the manuscript
    tex = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
           r"cohort & $n$ & $\hat\rho_1^2$ & $\bar\rho_1^2$ & "
           r"$\mathcal{R}_n$ & $\mathcal{I}_n$ & $\bar{\mathcal{I}}_n$ & "
           r"$\mathrm{AUC}_{\max}$\\",
           r"\midrule"]
    for r in rows:
        tb = r["I_total_ucl_bits"]
        tex.append(f"{r['label']} & {r['n']} & {r['rho2_fit']:.3f} & "
                   f"{r['rho2_ucl95']:.3f} & {r['R_attainable_ucl']:.3f} & "
                   f"{r['I_attainable_bits_ucl']:.2f} & "
                   f"{'--' if tb is None else format(tb, '.2f')} & "
                   f"{r['auc_ceiling_ucl']:.2f}\\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "attainable_summary_table.tex").write_text("\n".join(tex) + "\n")
    print(f"\nwrote {OUT/'attainable_summary.pdf'} and "
          f"{OUT/'attainable_summary_table.tex'}")


if __name__ == "__main__":
    main()
