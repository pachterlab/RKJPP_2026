"""Simulation check of the attainable-information ceiling.

Plants a linear--Gaussian channel with known canonical correlations, then, over a
grid of training-set sizes, trains several model families to predict the leading
genomic canonical score from imaging and evaluates them on a large held-out set.

The claim under test is two-sided:
  (a) no trained model exceeds  R_n = rho^2 n / (n + nu),  nu = d* (1-rho^2)/rho^2;
  (b) the best model tracks it, so the bound is tight rather than vacuous.

Writes notebooks/figures/synthetic/attainable_bound.pdf and a JSON summary.
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

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from rgit.bounds import (
    attainable_recoverability,
    attainable_information,
    channel_information,
    learning_cost,
    sample_size_for_fraction,
)

OUT = Path("notebooks/figures/synthetic")
OUT.mkdir(parents=True, exist_ok=True)

RHO2 = np.array([0.60, 0.35, 0.15, 0.05])  # planted channel spectrum
D_STAR = 20                                # imaging working dimension
N_TEST = 12000
N_GRID = [20, 30, 45, 65, 95, 140, 200, 300, 450, 700, 1100, 1700, 2600, 4000]
N_REPS = 40
SEED = 0


def draw_basis(rng, d_star=D_STAR):
    """A random orthogonal imaging basis, shared by a replicate's train and test.

    Re-drawing it per replicate is what makes this a fair test of the theorem:
    the ceiling is a Bayes risk under an isotropic (SNR-matched g-) prior on the
    regression vector, so the signal direction must be random, not fixed. Held
    fixed, a learner could in principle exploit the particular geometry and beat
    an average-case bound.
    """
    Q, _ = np.linalg.qr(rng.standard_normal((d_star, d_star)))
    return Q


def sample_channel(n, rng, Q, rho2=RHO2, d_star=D_STAR):
    """Draw (S, X) from the canonical form of the linear-Gaussian model.

    S_i are the genomic canonical scores (unit variance, independent); X is
    whitened imaging with X_i = rho_i S_i + sqrt(1-rho_i^2) e_i on the first k
    coordinates and pure noise elsewhere, then rotated by ``Q``. Any invertible
    linear reparametrization of either modality leaves the canonical
    correlations -- and therefore the bound -- unchanged, so this is the general
    case.
    """
    k = len(rho2)
    S = rng.standard_normal((n, k))
    X = rng.standard_normal((n, d_star))
    rho = np.sqrt(rho2)
    X[:, :k] = rho * S + np.sqrt(1.0 - rho2) * X[:, :k]
    return S, X @ Q.T


def r2(y_true, y_pred):
    ss = np.sum((y_true - y_pred) ** 2)
    return 1.0 - ss / np.sum((y_true - y_true.mean()) ** 2)


def model_zoo(n):
    alphas = np.logspace(-3, 4, 40)
    zoo = {
        "OLS": LinearRegression(),
        "ridge (CV)": RidgeCV(alphas=alphas),
        "kernel ridge (RBF)": KernelRidge(kernel="rbf", alpha=1.0, gamma=1.0 / D_STAR),
        "random forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, random_state=0, n_jobs=-1
        ),
        "grad. boosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, random_state=0
        ),
    }
    if n >= 60:
        zoo["MLP"] = MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=2000, alpha=1e-2,
            random_state=0, early_stopping=True,
        )
    return zoo


def main():
    names = list(model_zoo(10_000).keys())
    curves = {m: {n: [] for n in N_GRID} for m in names}
    insample = {n: [] for n in N_GRID}

    for rep in range(N_REPS):
        rr = np.random.default_rng(SEED + 1000 + rep)
        Q = draw_basis(rr)
        S_te, X_te = sample_channel(N_TEST, rr, Q)
        y_te = S_te[:, 0]
        for n in N_GRID:
            S_tr, X_tr = sample_channel(n, rr, Q)
            y_tr = S_tr[:, 0]
            sc = StandardScaler().fit(X_tr)
            Xtr, Xte = sc.transform(X_tr), sc.transform(X_te)
            for name, mdl in model_zoo(n).items():
                try:
                    mdl.fit(Xtr, y_tr)
                    curves[name][n].append(r2(y_te, mdl.predict(Xte)))
                except Exception:
                    curves[name][n].append(np.nan)
            # the in-sample number a naive analysis would report
            ols = LinearRegression().fit(Xtr, y_tr)
            insample[n].append(r2(y_tr, ols.predict(Xtr)))

    # ---------------- bound ----------------
    rho2_1 = RHO2[0]
    nu1 = float(learning_cost(rho2_1, D_STAR)[0])
    n_dense = np.logspace(np.log10(15), np.log10(6000), 300)
    bound = np.array(
        [attainable_recoverability(rho2_1, n, D_STAR)[0] for n in n_dense]
    )

    summary = {
        "planted_rho2": RHO2.tolist(),
        "d_star": D_STAR,
        "nu_leading": nu1,
        "channel_information_bits": channel_information(RHO2),
        "n_for_90pct": float(sample_size_for_fraction(rho2_1, D_STAR, 0.9)[0]),
        "attainable_information_bits": {
            str(n): attainable_information(RHO2, n, D_STAR) for n in N_GRID
        },
        "models": {},
        "violations": [],
    }
    # The theorem bounds the *expected* held-out R^2 over training draws, so the
    # test is on the mean with its Monte-Carlo standard error, not on a
    # per-replicate quantile (individual lucky draws may sit above the mean).
    for m in names:
        vals = [np.asarray(curves[m][n], dtype=float) for n in N_GRID]
        mean = [float(np.nanmean(v)) for v in vals]
        se = [float(np.nanstd(v, ddof=1) / np.sqrt(np.sum(np.isfinite(v))))
              for v in vals]
        summary["models"][m] = {
            "n": N_GRID, "mean_r2": mean, "se_r2": se,
            "median_r2": [float(np.nanmedian(v)) for v in vals],
        }
        for n, mu, s in zip(N_GRID, mean, se):
            b = float(attainable_recoverability(rho2_1, n, D_STAR)[0])
            if np.isfinite(mu) and mu - 2.0 * s > b:  # excess beyond 2 s.e.
                summary["violations"].append(
                    {"model": m, "n": n, "mean_r2": mu, "se": s, "bound": b,
                     "excess_sigma": (mu - b) / s})
    summary["in_sample_ols_mean_r2"] = [
        float(np.mean(insample[n])) for n in N_GRID
    ]

    # ---------------- figure ----------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    ax.semilogx(n_dense, bound, "k-", lw=2.4, zorder=5,
                label=r"attainable ceiling $\mathcal{R}_n$")
    ax.axhline(rho2_1, color="k", ls=":", lw=1.6,
               label=r"channel ceiling $\rho_1^2$ ($n=\infty$)")
    ax.plot(N_GRID, summary["in_sample_ols_mean_r2"], color="crimson", ls="--",
            marker="v", ms=4, lw=1.2, label="in-sample OLS (what a naive fit reports)")
    for m in names:
        ax.plot(N_GRID, summary["models"][m]["mean_r2"], marker="o", ms=3.5,
                lw=1.3, alpha=0.85, label=m)
    ax.axvline(nu1, color="grey", ls="-.", lw=1.0)
    ax.annotate(r"$n=\nu$", xy=(nu1, 0.0), xytext=(10, 6),
                textcoords="offset points", color="grey", fontsize=9)
    ax.set_xlabel("training patients $n$")
    ax.set_ylabel(r"held-out $R^2$ of leading genomic direction")
    ax.set_ylim(-0.25, min(1.02, rho2_1 * 1.8))
    ax.set_title(r"(a) no algorithm crosses $\mathcal{R}_n$")
    ax.legend(fontsize=7.2, loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.25)

    ax = axes[1]
    In = np.array([attainable_information(RHO2, n, D_STAR) for n in n_dense])
    ax.semilogx(n_dense, In, "k-", lw=2.4, label=r"$\mathcal{I}_n$ (all directions)")
    ax.axhline(channel_information(RHO2), color="k", ls=":", lw=1.6,
               label=r"$I(G;X)$ channel information")
    for j, r2j in enumerate(RHO2):
        ax.semilogx(
            n_dense,
            [-0.5 * np.log2(1 - attainable_recoverability(r2j, n, D_STAR)[0])
             for n in n_dense],
            lw=1.1, alpha=0.8,
            label=rf"direction {j+1} ($\rho^2={r2j:g}$)",
        )
    for n in (190, 129, 702):  # the three cohort sizes in the manuscript
        ax.axvline(n, color="grey", lw=0.7, alpha=0.5)
    ax.set_xlabel("cohort size $n$")
    ax.set_ylabel("attainable information (bits / patient)")
    ax.set_title("(b) the information ceiling is a function of cohort size")
    ax.legend(fontsize=7.2, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT / "attainable_bound.pdf", bbox_inches="tight")
    fig.savefig(OUT / "attainable_bound.png", dpi=160, bbox_inches="tight")

    (OUT / "attainable_bound.json").write_text(json.dumps(summary, indent=2))

    print(f"planted rho1^2 = {rho2_1}, d* = {D_STAR}, nu = {nu1:.1f}")
    print(f"channel I(G;X) = {summary['channel_information_bits']:.3f} bits")
    print(f"n for 90% of ceiling = {summary['n_for_90pct']:.0f}")
    print("\n  n   bound   " + "  ".join(f"{m[:11]:>11s}" for m in names))
    for i, n in enumerate(N_GRID):
        b = attainable_recoverability(rho2_1, n, D_STAR)[0]
        row = "  ".join(f"{summary['models'][m]['median_r2'][i]:11.3f}" for m in names)
        print(f"{n:5d} {b:7.3f}   {row}")
    n_cells = sum(len(v["n"]) for v in summary["models"].values())
    summary["n_cells_tested"] = n_cells
    summary["expected_false_exceedances"] = 0.023 * n_cells  # one-sided 2 s.e.
    print(f"\nexceedances of the bound (mean > bound + 2 s.e.): "
          f"{len(summary['violations'])} of {n_cells} model x n cells "
          f"(expected by chance alone: {summary['expected_false_exceedances']:.1f})")
    for v in summary["violations"]:
        print("   ", v)
    print(f"\nwrote {OUT/'attainable_bound.pdf'}")


if __name__ == "__main__":
    main()
