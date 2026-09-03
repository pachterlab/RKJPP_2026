"""Raw versus demographically deconfounded recoverability spectrum.

Redraws the ``deconfound_comparison`` figure of the companion notebook from the
``stats.json`` the notebook writes, so the figure can be restyled without
re-running the cohort pipeline. One bar pair per canonical component: the
cross-validated recoverability on the raw modeling matrices and on the same
matrices with the cohort's confounders residualized from both modalities.

    python scripts/deconfound_figure.py adni            # one or more cohorts
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

from attainable_bound_cohorts import FIGDIR, OUTROOT  # noqa: E402  (same directory)


def figure(name: str):
    figdir = OUTROOT / FIGDIR[name]
    stats = json.load(open(figdir / "stats.json"))
    dc = stats["deconfound_comparison"]
    raw, dec = dc["raw"], dc["deconfounded"]
    K = min(5, len(raw["cv_R"]), len(dec["cv_R"]))
    x = np.arange(1, K + 1)
    w = 0.38

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.bar(x - w / 2, raw["cv_R"][:K], w, color="#4C72B0",
           label=f"raw (eff. rank {raw['eff_rank']})")
    ax.bar(x + w / 2, dec["cv_R"][:K], w, color="#DD8452",
           label=f"deconfounded on {', '.join(dc['confounders'])} "
                 f"(eff. rank {dec['eff_rank']})")
    ax.set_xlabel("canonical component $i$")
    ax.set_ylabel(r"cross-validated $R_i$")
    ax.set_xticks(x)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(figdir / "deconfound_comparison.pdf", bbox_inches="tight")
    fig.savefig(figdir / "deconfound_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {figdir/'deconfound_comparison.pdf'}")


if __name__ == "__main__":
    for c in (sys.argv[1:] or ["adni"]):
        figure(c)
