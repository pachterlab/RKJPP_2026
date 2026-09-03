"""Figures for the TCGA-KIRC target-class and four-representation analyses.

Reads
  notebooks/figures/kirc_target_classes.json                 (scripts/kirc_target_classes.py)
  notebooks/figures/kirc_representations/summary.json        (scripts/build_kirc_representation_notebook.py)
and writes
  notebooks/figures/kirc_target_classes.{pdf,png}      main-text figure: decodability by target class
  notebooks/figures/kirc_representations.{pdf,png}     appendix figure: variance budget and spectra
  notebooks/figures/kirc_target_classes_table.tex      appendix table: every target x representation

    python scripts/kirc_target_classes_figure.py
"""

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from rgit.figures import label_panels  # noqa: E402
FIG = REPO / "notebooks/figures"

targets = json.load(open(FIG / "kirc_target_classes.json"))["targets"]
summary = json.load(open(FIG / "kirc_representations/summary.json"))

IMAGING = ["tumor_radiomics", "organ_radiomics", "tumor_radimagenet", "whole_radimagenet"]
PRETTY = {"tumor_radiomics": "Tumor radiomics", "organ_radiomics": "Kidney radiomics",
          "tumor_radimagenet": "Tumor deep", "whole_radimagenet": "Whole-volume deep"}
SHORT = {"tumor_radiomics": "Tumor\nradiomics", "organ_radiomics": "Kidney\nradiomics",
         "tumor_radimagenet": "Tumor\ndeep", "whole_radimagenet": "Whole-vol.\ndeep"}
# All-pairs CVD-validated categorical slots (validated with the dataviz palette checker).
COLORS = {"tumor_radiomics": "#2a78d6", "organ_radiomics": "#eb6834",
          "tumor_radimagenet": "#1baf7a", "whole_radimagenet": "#4a3aa7"}
INK, MUTED, GRID, ACCENT, DEEMPH = "#0b0b0b", "#52514e", "#d8d7d0", "#2a78d6", "#b8b7b2"

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42,
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
})

AUC_ROWS = [  # (key, label, class)
    ("stage_III_IV", "stage III-IV", "control"),
    ("sex_female", "sex", "control"),
    ("mut_VHL", "VHL", "mutation"),
    ("mut_PBRM1", "PBRM1", "mutation"),
    ("mut_BAP1", "BAP1", "mutation"),
    ("mut_SETD2", "SETD2", "mutation"),
]
R2_ROWS = [
    ("age", "age", "control"),
    ("angiogenesis", "angiogenesis", "signature"),
    ("myeloid_inflammation", "myeloid inflammation", "signature"),
    ("t_effector", "T-effector", "signature"),
]


def best(key):
    t = targets[key]
    b = t["best_image"]
    return b, t["by_image"][b]


# ---------------------------------------------------------------- main-text figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

fig, (axA, axB) = plt.subplots(1, 2, figsize=(5.5, 2.3), gridspec_kw={"width_ratios": [1.15, 1]})

# (a) AUC targets: observed (best representation) against the null 95th percentile
ys = np.arange(len(AUC_ROWS))[::-1]
for y, (key, label, cls) in zip(ys, AUC_ROWS):
    rep, b = best(key)
    sig = b["p"] < 0.05
    axA.plot([0.5, b["null_q95"]], [y, y], color=DEEMPH, lw=3.5, solid_capstyle="butt", zorder=2)
    axA.plot(b["observed"], y, "o", ms=5, color=ACCENT if sig else "white",
             markeredgecolor=ACCENT if sig else MUTED, markeredgewidth=1.0, zorder=4)
    axA.annotate(f"{b['observed']:.2f}", (max(b["observed"], b["null_q95"]), y), textcoords="offset points",
                 xytext=(5, 0), ha="left", va="center", fontsize=6.5)
axA.set_yticks(ys)
axA.set_yticklabels([r"$\it{" + l + "}$" if c == "mutation" else l for _, l, c in AUC_ROWS])
axA.set_xlim(0.5, 1.0)
axA.set_ylim(-0.7, 6.2)
axA.set_xlabel("cross-validated AUC")
axA.grid(axis="y", visible=False)
axA.axhline(3.5, color=GRID, lw=0.8)
axA.text(0.995, 6.15, "positive controls", fontsize=6, color=MUTED, va="top", ha="right")
axA.text(0.995, 3.4, "driver mutation status", fontsize=6, color=MUTED, va="top", ha="right")

# (b) R^2 targets: raw (best representation) and after demographic + scanner adjustment
ys = np.arange(len(R2_ROWS))[::-1]
h = 0.34
for y, (key, label, cls) in zip(ys, R2_ROWS):
    rep, b = best(key)
    sig = b["p"] < 0.05
    col = ACCENT if sig else DEEMPH
    axB.barh(y + h / 2, max(b["observed"], 0), h, color=col, zorder=3)
    axB.annotate(f"{b['observed']:.3f}", (max(b["observed"], 0), y + h / 2),
                 textcoords="offset points", xytext=(3, 0), va="center", fontsize=6.5)
    if "deconf_all_r2" in targets[key]:
        v = max(targets[key]["deconf_all_r2"], 0)
        axB.barh(y - h / 2, v, h, color=col, alpha=0.45, zorder=3)
        axB.annotate(f"{targets[key]['deconf_all_r2']:.3f}", (v, y - h / 2),
                     textcoords="offset points", xytext=(3, 0), va="center", fontsize=6.5)
    axB.plot([b["null_q95"]] * 2, [y - h, y + h], color=INK, lw=0.9, zorder=4)
axB.set_yticks(ys)
axB.set_yticklabels([l for _, l, _ in R2_ROWS])
axB.set_xlim(0, 0.19)
axB.set_ylim(-0.7, 4.1)
axB.set_xlabel(r"cross-validated $R^2$")
axB.grid(axis="y", visible=False)
axB.axhline(2.5, color=GRID, lw=0.8)
axB.text(0.188, 4.05, "positive control", fontsize=6, color=MUTED, va="top", ha="right")
axB.text(0.188, 2.42, "IMmotion151 signatures", fontsize=6, color=MUTED, va="top", ha="right")

handles = [
    Line2D([], [], color=DEEMPH, lw=3.5, label="permutation null (95th percentile)"),
    Line2D([], [], marker="o", ls="", ms=5, color=ACCENT, label="above null ($p<0.05$)"),
    Line2D([], [], marker="o", ls="", ms=5, color="white", markeredgecolor=MUTED, label="within null"),
    Patch(color=ACCENT, label="raw"),
    Patch(color=ACCENT, alpha=0.45, label="age, sex, ethnicity, race and scanner adjusted"),
    Line2D([], [], color=INK, lw=0.9, label="null (95th percentile)"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, handlelength=1.6,
           columnspacing=1.2, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(w_pad=1.5, rect=(0, 0.13, 1, 1))
label_panels(fig, (axA, axB))
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"kirc_target_classes.{ext}", bbox_inches="tight")
print("saved", FIG / "kirc_target_classes.pdf")

# ---------------------------------------------------------------- appendix figure
budget, spectrum, spec_null = summary["budget"], summary["spectrum"], summary["spectrum_null_q95"]
bounds = summary["bounds"]
N_AXES = 12
fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(5.5, 2.2), gridspec_kw={"width_ratios": [1, 1.2, 1]})
yr = np.arange(4)[::-1]

tau = [100 * budget[m]["tau"] for m in IMAGING]
axA.barh(yr, tau, 0.62, color=[COLORS[m] for m in IMAGING], zorder=3)
for y, v in zip(yr, tau):
    axA.annotate(f"{v:.2f}%", (v, y), textcoords="offset points", xytext=(3, 0), va="center", fontsize=6.5)
axA.set_yticks(yr)
axA.set_yticklabels([PRETTY[m] for m in IMAGING])
axA.set_xlabel("transcriptome variance\nrecovered (%)")
axA.set_xlim(0, max(tau) * 1.35)
axA.grid(axis="y", visible=False)

xs = np.arange(1, N_AXES + 1)
axB.plot(xs, np.mean([spec_null[m] for m in IMAGING], axis=0), color=MUTED, lw=0.9, ls="--", zorder=2)
for m in IMAGING:
    axB.plot(xs, spectrum[m][:N_AXES], color=COLORS[m], lw=1.6, zorder=3)
axB.set_xlabel("canonical axis")
axB.set_ylabel(r"cross-validated $\hat R_i^{\mathrm{cv}}$")
axB.set_xticks([1, 3, 5, 7, 9, 11])
axB.axhline(0, color=MUTED, lw=0.6)

raw = [bounds[m]["R1_cv"] for m in IMAGING]
adj = [bounds[m]["deconf_all"]["R1_cv"] for m in IMAGING]
axC.barh(yr + 0.19, raw, 0.36, color=[COLORS[m] for m in IMAGING], zorder=3)
axC.barh(yr - 0.19, adj, 0.36, color=[COLORS[m] for m in IMAGING], alpha=0.4, zorder=3)
for y, m in zip(yr, IMAGING):
    axC.plot([bounds[m]["R1_cv_null_q95"]] * 2, [y - 0.4, y + 0.4], color=INK, lw=0.8, zorder=4)
axC.set_yticks(yr)
axC.set_yticklabels([])
axC.set_xlabel(r"leading axis $\hat R_1^{\mathrm{cv}}$")
axC.set_xlim(0, 0.45)
axC.grid(axis="y", visible=False)

handles = [Line2D([], [], color=COLORS[m], lw=1.6, label=PRETTY[m]) for m in IMAGING] + [
    Line2D([], [], color=MUTED, lw=0.9, ls="--", label="permutation null (95th percentile), panel b"),
    Patch(color=MUTED, label="raw, panel c"),
    Patch(color=MUTED, alpha=0.4, label="adjusted, panel c"),
    Line2D([], [], color=INK, lw=0.8, label="null (95th percentile), panel c"),
]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, handlelength=1.6,
           columnspacing=1.2, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(w_pad=1.2, rect=(0, 0.16, 1, 1))
label_panels(fig, (axA, axB, axC))
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"kirc_representations.{ext}", bbox_inches="tight")
print("saved", FIG / "kirc_representations.pdf")

# ---------------------------------------------------------------- appendix table
rows = []
for key, label, cls in AUC_ROWS + R2_ROWS:
    t = targets[key]
    metric = "AUC" if t["kind"] in ("mutation", "control_binary") else "$R^2$"
    n = f"{t['n_mutated']}/{t['n']}" if cls == "mutation" else str(t["n"])
    cells = []
    for m in IMAGING:
        b = t["by_image"][m]
        v = f"{b['observed']:.2f}" if metric == "AUC" else f"{b['observed']:.3f}"
        if b["p"] < 0.05:
            v = r"\textbf{" + v + "}"
        cells.append(v)
    adjc = f"{t['deconf_all_r2']:.3f}" if "deconf_all_r2" in t else "--"
    lab = r"\emph{" + label + "}" if cls == "mutation" else label
    rows.append(f"{lab} & {n} & {metric} & " + " & ".join(cells) + f" & {adjc}\\\\")
tex = "\n".join([
    r"\begin{tabular}{llr" + "r" * 4 + "r}",
    r"\toprule",
    r"target & $n$ & metric & tumor radiomics & kidney radiomics & tumor deep & whole-vol.\ deep & adjusted\\",
    r"\midrule",
    *rows,
    r"\bottomrule",
    r"\end{tabular}",
])
(FIG / "kirc_target_classes_table.tex").write_text(tex + "\n")
print("saved", FIG / "kirc_target_classes_table.tex")
