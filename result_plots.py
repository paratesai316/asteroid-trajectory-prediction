"""
results_plots.py
================
Generates all publication-quality figures for the Results section.

Run this script after your full 45-asteroid run is complete.
It expects the following files in the directories you specify:

  data/split_study/split_study_<asteroid>.csv   (one per asteroid)
  data/mc_summary/mc_summary_<asteroid>.csv     (one per asteroid)

Usage:
  python results_plots.py

All figures are saved as high-resolution PNG and PDF to output/figures/.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import LogLocator, NullFormatter
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
SPLIT_DIR  = Path("output/split_study")
MC_DIR     = Path("output/monte_carlo")
FIG_DIR    = Path("output/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

AU_TO_KM   = 1.495978707e8
DPI        = 200
FIGFMT     = ["png", "pdf"]

PALETTE = {
    "10%":  "#e74c3c",
    "20%":  "#e67e22",
    "30%":  "#2ecc71",
    "40%":  "#3498db",
    "50%":  "#9b59b6",
}
SPLIT_COLORS = list(PALETTE.values())


def save_fig(fig, name):
    for fmt in FIGFMT:
        fig.savefig(FIG_DIR / f"{name}.{fmt}", dpi=DPI,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved -> {name}.png / .pdf")


def load_all_splits():
    """Load split study CSVs for all available asteroids."""
    frames = []
    for f in sorted(SPLIT_DIR.glob("split_study_*.csv")):
        name = f.stem.replace("split_study_", "")
        df   = pd.read_csv(f)
        df["asteroid"] = name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No split study files found in {SPLIT_DIR}")
    return pd.concat(frames, ignore_index=True)


def load_all_mc():
    """Load MC summary CSVs for all available asteroids."""
    summaries = {}
    for f in sorted(MC_DIR.glob("mc_summary_*.csv")):
        name = f.stem.replace("mc_summary_", "")
        summaries[name] = pd.read_csv(f)
    if not summaries:
        raise FileNotFoundError(f"No MC summary files found in {MC_DIR}")
    return summaries


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Split study: RMSE vs split fraction, all asteroids
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_rmse_vs_split(df_all):
    """
    Line plot: RMSE (AU, log scale) vs split percentage.
    One line per asteroid, colour-coded by orbital class if available,
    grey otherwise. Mean trend shown as a thick black line.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Figure 1 — Fit Window Sensitivity: RMSE vs Split Fraction\n"
                 "17-body model, GR + Yarkovsky, DOP853 rtol=1e-13",
                 fontsize=13, fontweight="bold")

    asteroids = df_all["asteroid"].unique()

    # ── Left panel: individual lines (log scale) ──────────────────────────────
    ax = axes[0]
    splits = sorted(df_all["split_pct"].unique())

    for ast in asteroids:
        sub = df_all[df_all["asteroid"] == ast].sort_values("split_pct")
        ax.semilogy(sub["split_pct"], sub["RMSE_AU"],
                    "o-", lw=0.8, ms=3, alpha=0.45, color="steelblue")

    # Mean across all asteroids
    mean_rmse = df_all.groupby("split_pct")["RMSE_AU"].median()
    ax.semilogy(mean_rmse.index, mean_rmse.values,
                "o-", lw=2.5, ms=8, color="black", label="Median (all asteroids)",
                zorder=10)

    ax.set_xlabel("Fitting window fraction (%)", fontsize=11)
    ax.set_ylabel("RMSE (AU)", fontsize=11)
    ax.set_title(f"RMSE vs fit fraction  ({len(asteroids)} asteroids)", fontsize=11)
    ax.set_xticks(splits)
    ax.set_xticklabels([f"{s}%" for s in splits])
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.yaxis.set_minor_locator(LogLocator(subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())

    # Annotate the 20→30% inflection
    ax.annotate("Yarkovsky\nconvergence\nthreshold",
                xy=(30, float(mean_rmse[30])),
                xytext=(35, float(mean_rmse[30]) * 8),
                fontsize=8, color="darkred",
                arrowprops=dict(arrowstyle="->", color="darkred", lw=1.2))

    # ── Right panel: improvement ratio relative to 10% ────────────────────────
    ax2 = axes[1]
    base = df_all[df_all["split_pct"] == 10].set_index("asteroid")["RMSE_AU"]

    for ast in asteroids:
        sub = df_all[df_all["asteroid"] == ast].sort_values("split_pct")
        # base[ast] can be a scalar or a Series if the asteroid appears
        # more than once — use .iloc[0] after filtering to always get a scalar
        b_series = df_all[
            (df_all["asteroid"] == ast) & (df_all["split_pct"] == 10)
        ]["RMSE_AU"]
        if b_series.empty:
            continue
        b = float(b_series.iloc[0])
        ratio = b / sub["RMSE_AU"].values
        ax2.semilogy(sub["split_pct"].values, ratio,
                     "o-", lw=0.8, ms=3, alpha=0.45, color="steelblue")

    mean_base = (
        df_all[df_all["split_pct"] == 10]
        .groupby("asteroid")["RMSE_AU"]
        .first()
    )
    mean_ratio = []
    for s in splits:
        sub_s = (
            df_all[df_all["split_pct"] == s]
            .groupby("asteroid")["RMSE_AU"]
            .first()
        )
        # Align on common asteroids only, compute ratio, take median
        common = mean_base.index.intersection(sub_s.index)
        if len(common) == 0:
            mean_ratio.append(np.nan)
        else:
            r = mean_base.loc[common] / sub_s.loc[common]
            mean_ratio.append(float(r.median()))

    ax2.semilogy(splits, mean_ratio, "o-", lw=2.5, ms=8,
                 color="black", label="Median improvement ratio", zorder=10)
    ax2.axhline(1, color="gray", ls="--", lw=0.8)
    ax2.set_xlabel("Fitting window fraction (%)", fontsize=11)
    ax2.set_ylabel("RMSE improvement vs 10% baseline (×)", fontsize=11)
    ax2.set_title("Prediction improvement relative to 10% fit fraction", fontsize=11)
    ax2.set_xticks(splits)
    ax2.set_xticklabels([f"{s}%" for s in splits])
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, which="both")

    save_fig(fig, "fig1_rmse_vs_split")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Split study: RMSE distribution (box plots per split)
# ═══════════════════════════════════════════════════════════════════════════════
def fig2_rmse_boxplots(df_all):
    """
    Box plots showing the distribution of RMSE across all 45 asteroids
    for each split fraction. Shows spread, median, and outliers.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 2 — Distribution of RMSE Across All Asteroids by Split Fraction",
                 fontsize=13, fontweight="bold")

    splits = sorted(df_all["split_pct"].unique())

    for ax, col, label, unit in [
        (axes[0], "RMSE_AU", "RMSE (AU)", "AU"),
        (axes[1], "RMSE_km", "RMSE (km)", "km"),
    ]:
        data = [df_all[df_all["split_pct"] == s][col].values for s in splits]
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color="black", lw=2))
        for patch, color in zip(bp["boxes"], SPLIT_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        if col == "RMSE_AU":
            ax.set_yscale("log")
        ax.set_xticklabels([f"{s}%" for s in splits])
        ax.set_xlabel("Fitting window fraction (%)", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f"RMSE distribution across {len(df_all['asteroid'].unique())} asteroids",
                     fontsize=10)
        ax.grid(axis="y", alpha=0.3, which="both")

    save_fig(fig, "fig2_rmse_boxplots")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Split study: A2 stability across splits
# ═══════════════════════════════════════════════════════════════════════════════
def fig3_A2_stability(df_all):
    """
    Scatter plot of estimated A2 values vs split fraction for all asteroids.
    Shows how A2 estimates stabilise as more data is used.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 3 — Yarkovsky A2 Estimate Stability vs Fitting Fraction",
                 fontsize=13, fontweight="bold")

    splits   = sorted(df_all["split_pct"].unique())
    asteroids = df_all["asteroid"].unique()

    ax = axes[0]
    for ast in asteroids:
        sub = df_all[df_all["asteroid"] == ast].sort_values("split_pct")
        ax.plot(sub["split_pct"], sub["A2"] * 1e15,
                "o-", lw=0.8, ms=3, alpha=0.4, color="steelblue")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Fitting window fraction (%)", fontsize=11)
    ax.set_ylabel("Estimated A₂ (×10⁻¹⁵ AU³/day²)", fontsize=11)
    ax.set_title("A₂ estimates — individual asteroids", fontsize=10)
    ax.set_xticks(splits)
    ax.set_xticklabels([f"{s}%" for s in splits])
    ax.grid(alpha=0.3)

    # Right panel: standard deviation of A2 across asteroids per split
    ax2 = axes[1]
    std_A2 = df_all.groupby("split_pct")["A2"].std() * 1e15
    rng_A2 = df_all.groupby("split_pct")["A2"].apply(
        lambda x: x.max() - x.min()) * 1e15
    ax2.plot(splits, std_A2.values, "o-", color="#e74c3c", lw=2,
             ms=7, label="Std dev of A₂ across asteroids")
    ax2.set_xlabel("Fitting window fraction (%)", fontsize=11)
    ax2.set_ylabel("Std dev of A₂ (×10⁻¹⁵ AU³/day²)", fontsize=11)
    ax2.set_title("A₂ estimation consistency — lower = more stable", fontsize=10)
    ax2.set_xticks(splits)
    ax2.set_xticklabels([f"{s}%" for s in splits])
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.annotate("Convergence\nbegins here",
                 xy=(30, float(std_A2[30])),
                 xytext=(35, float(std_A2[30]) * 1.3),
                 fontsize=8.5, color="darkred",
                 arrowprops=dict(arrowstyle="->", color="darkred"))

    save_fig(fig, "fig3_A2_stability")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Single asteroid split study (detailed, e.g. Toutatis)
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_single_asteroid_split(df_all, asteroid="toutatis"):
    """
    Detailed 4-panel split study result for one representative asteroid.
    """
    sub = df_all[df_all["asteroid"] == asteroid].sort_values("split_pct")
    if sub.empty:
        print(f"  No data for asteroid '{asteroid}', skipping fig4")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Figure 4 — Split Study Detail: {asteroid.capitalize()}\n"
                 "17-body N-body | DOP853 rtol=1e-13 | GR + Yarkovsky",
                 fontsize=13, fontweight="bold")

    pcts   = sub["split_pct"].values
    colors = SPLIT_COLORS[:len(pcts)]

    # Panel 1: RMSE (AU) bar chart
    ax = axes[0, 0]
    bars = ax.bar([f"{p}%" for p in pcts], sub["RMSE_AU"].values,
                  color=colors, alpha=0.88, edgecolor="white")
    ax.set_yscale("log")
    ax.set_ylabel("RMSE (AU)")
    ax.set_title("RMSE vs fit fraction")
    ax.grid(axis="y", alpha=0.3, which="both")
    for bar, val in zip(bars, sub["RMSE_km"].values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() * 1.6,
                f"{val:.3f} km", ha="center", va="bottom", fontsize=8.5)

    # Panel 2: Max and Mean error (km)
    ax = axes[0, 1]
    x  = np.arange(len(pcts)); w = 0.35
    ax.bar(x - w/2, sub["Max_km"].values,  w, color="#e74c3c",
           alpha=0.85, label="Max error")
    ax.bar(x + w/2, sub["Mean_km"].values, w, color="#3498db",
           alpha=0.85, label="Mean error")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}%" for p in pcts])
    ax.set_ylabel("Error (km)")
    ax.set_title("Max and mean prediction error")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Panel 3: Prediction window size vs RMSE scatter
    ax = axes[1, 0]
    sc = ax.scatter(sub["pred_rows"].values / 24,  # convert hours to days
                    sub["RMSE_km"].values,
                    c=pcts, cmap="RdYlGn_r", s=120,
                    edgecolors="black", linewidth=0.5, zorder=5)
    plt.colorbar(sc, ax=ax, label="Fit fraction (%)")
    for _, row in sub.iterrows():
        ax.annotate(f"{int(row.split_pct)}%",
                    (row.pred_rows/24, row.RMSE_km),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Prediction window length (days)")
    ax.set_ylabel("RMSE (km)")
    ax.set_title("Prediction accuracy vs window length")
    ax.grid(alpha=0.3)

    # Panel 4: A2 evolution
    ax = axes[1, 1]
    ax.plot(pcts, sub["A2"].values * 1e15, "o-",
            color="darkorange", lw=2, ms=8, markeredgecolor="black")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Fitting window fraction (%)")
    ax.set_ylabel("Estimated A₂ (×10⁻¹⁵ AU³/day²)")
    ax.set_title("Yarkovsky A₂ estimate stability")
    ax.set_xticks(pcts)
    ax.set_xticklabels([f"{p}%" for p in pcts])
    ax.grid(alpha=0.3)
    ax.annotate("Sign changes →\nunder-constrained arc",
                xy=(10, float(sub[sub.split_pct==10]["A2"].values[0]*1e15)),
                xytext=(25, float(sub[sub.split_pct==10]["A2"].values[0]*1e15)+30),
                fontsize=7.5, color="darkred",
                arrowprops=dict(arrowstyle="->", color="darkred"))

    plt.tight_layout()
    save_fig(fig, f"fig4_split_detail_{asteroid}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — MC: uncertainty propagation for single asteroid
# ═══════════════════════════════════════════════════════════════════════════════
def fig5_mc_single(mc_summaries, asteroid="toutatis"):
    if asteroid not in mc_summaries:
        print(f"  No MC data for {asteroid}, skipping fig5")
        return

    df = mc_summaries[asteroid]
    t_day = (df["julian_date"] - df["julian_date"].iloc[0]) * 24 / 24  # days

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Figure 5 — Monte Carlo Uncertainty Propagation: {asteroid.capitalize()}\n"
        "50 samples | 1 arcsec position noise | 0.1 m/s velocity noise | N=17",
        fontsize=13, fontweight="bold"
    )

    # Panel 1: Uncertainty bands (distance from nominal)
    ax = axes[0]
    p16_km  = df["p16_au"] * AU_TO_KM
    p84_km  = df["p84_au"] * AU_TO_KM
    p025_km = df["p025_au"] * AU_TO_KM
    p975_km = df["p975_au"] * AU_TO_KM
    p50_km  = df["p50_au"] * AU_TO_KM
    nom_km  = df["nominal_err_km"]

    ax.fill_between(t_day, p025_km, p975_km,
                    alpha=0.12, color="#3498db", label="2σ (p2.5–p97.5)")
    ax.fill_between(t_day, p16_km,  p84_km,
                    alpha=0.30, color="#3498db", label="1σ (p16–p84)")
    ax.semilogy(t_day, np.maximum(p50_km,  1e-6), color="#2980b9",
                lw=1.2, label="MC median spread")
    ax.semilogy(t_day, np.maximum(nom_km, 1e-6),  color="#e74c3c",
                lw=1.5, ls="--", label="Nominal model error")

    ax.set_xlabel("Days from start of prediction window", fontsize=10)
    ax.set_ylabel("Distance from nominal trajectory (km)", fontsize=10)
    ax.set_title("Uncertainty growth from initial state noise", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, which="both")

    # Panel 2: sigma1 (1-sigma half-width) over time
    ax = axes[1]
    s1 = df["sigma1_au"] * AU_TO_KM
    s2 = df["sigma2_au"] * AU_TO_KM
    s3 = df["sigma3_au"] * AU_TO_KM
    ax.semilogy(t_day, np.maximum(s1, 1e-6), color="#e74c3c", lw=1.5,
                label="1σ half-width")
    ax.semilogy(t_day, np.maximum(s2, 1e-6), color="#e67e22", lw=1.2,
                ls="--", label="2σ half-width")
    ax.semilogy(t_day, np.maximum(s3, 1e-6), color="#9b59b6", lw=1.0,
                ls=":", label="3σ half-width")
    ax.set_xlabel("Days from start of prediction window", fontsize=10)
    ax.set_ylabel("Uncertainty half-width (km)", fontsize=10)
    ax.set_title("σ-band growth over prediction window", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # Panel 3: Ratio sigma1 / nominal error
    ax = axes[2]
    ratio = df["sigma1_au"] / df["nominal_err_au"].replace(0, np.nan)
    ax.semilogy(t_day, ratio, color="darkorange", lw=1.5,
                label="σ₁ / Nominal RMSE")
    ax.axhline(1, color="gray", ls="--", lw=0.8,
               label="Ratio = 1 (equal contributions)")
    ax.set_xlabel("Days from start of prediction window", fontsize=10)
    ax.set_ylabel("σ₁ / Nominal model error (dimensionless)", fontsize=10)
    ax.set_title("Obs. uncertainty vs model error dominance", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    ax.annotate(f"Final ratio: {float(ratio.iloc[-1]):.0f}×\n"
                "→ obs. noise dominates",
                xy=(float(t_day.iloc[-1]), float(ratio.iloc[-1])),
                xytext=(float(t_day.iloc[-1]) * 0.6, float(ratio.iloc[-1]) * 0.3),
                fontsize=8, color="darkred",
                arrowprops=dict(arrowstyle="->", color="darkred"))

    plt.tight_layout()
    save_fig(fig, f"fig5_mc_uncertainty_{asteroid}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — MC: final sigma1 across all asteroids (bar chart)
# ═══════════════════════════════════════════════════════════════════════════════
def fig6_mc_comparison_all(mc_summaries):
    """
    Bar chart comparing final 1-sigma uncertainty and nominal RMSE
    for all available asteroids.
    """
    names   = []
    sig1_km = []
    nom_km  = []

    for name, df in sorted(mc_summaries.items()):
        names.append(name)
        sig1_km.append(float(df["sigma1_au"].iloc[-1]) * AU_TO_KM)
        nom_km.append(float(df["nominal_err_au"].iloc[-1]) * AU_TO_KM)

    n = len(names)
    if n == 0:
        print("  No MC data found, skipping fig6")
        return

    fig, axes = plt.subplots(2, 1, figsize=(max(14, n * 0.45), 11))
    fig.suptitle("Figure 6 — Monte Carlo Results: All Asteroids\n"
                 "Final 1σ uncertainty and nominal model RMSE at end of prediction window",
                 fontsize=13, fontweight="bold")

    x  = np.arange(n)
    w  = 0.38

    # Panel 1: side-by-side bars (log scale)
    ax = axes[0]
    ax.bar(x - w/2, sig1_km, w, color="#3498db", alpha=0.85,
           label="1σ uncertainty (obs. noise propagation)")
    ax.bar(x + w/2, nom_km,  w, color="#e74c3c", alpha=0.85,
           label="Nominal model RMSE")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("Distance (km, log scale)")
    ax.set_title("1σ uncertainty vs nominal RMSE — log scale")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, which="both")

    # Panel 2: ratio (sigma1 / nominal) — how much obs noise dominates
    ax = axes[1]
    ratio = [s/n if n > 0 else np.nan
             for s, n in zip(sig1_km, nom_km)]
    bars  = ax.bar(x, ratio, color="#9b59b6", alpha=0.85)
    ax.set_yscale("log")
    ax.axhline(1, color="gray", ls="--", lw=0.8,
               label="Ratio = 1 (equal contributions)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("σ₁ / Nominal RMSE (dimensionless, log scale)")
    ax.set_title("Dominance of observational uncertainty over model error\n"
                 "(ratio >> 1 means better telescopes matter more than better physics)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, which="both")

    plt.tight_layout()
    save_fig(fig, "fig6_mc_all_asteroids")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Overall model accuracy: RMSE heatmap (asteroids × splits)
# ═══════════════════════════════════════════════════════════════════════════════
def fig7_heatmap(df_all):
    """
    Heatmap: asteroids (rows) × split fractions (columns),
    colour = log10(RMSE_km). Quickly shows which asteroids are hardest to
    predict and how prediction quality varies with data volume.
    """
    splits    = sorted(df_all["split_pct"].unique())
    asteroids = sorted(df_all["asteroid"].unique())

    matrix = np.full((len(asteroids), len(splits)), np.nan)
    for i, ast in enumerate(asteroids):
        for j, s in enumerate(splits):
            row = df_all[(df_all["asteroid"] == ast) &
                         (df_all["split_pct"] == s)]
            if not row.empty:
                matrix[i, j] = np.log10(float(row["RMSE_km"].values[0]))

    fig, ax = plt.subplots(figsize=(9, max(8, len(asteroids) * 0.28)))
    fig.suptitle("Figure 7 — RMSE Heatmap: All Asteroids × All Split Fractions\n"
                 "Colour = log₁₀(RMSE in km)",
                 fontsize=12, fontweight="bold")

    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r",
                   vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))
    plt.colorbar(im, ax=ax, label="log₁₀(RMSE / km)")

    ax.set_xticks(range(len(splits)))
    ax.set_xticklabels([f"{s}%" for s in splits], fontsize=10)
    ax.set_yticks(range(len(asteroids)))
    ax.set_yticklabels(asteroids, fontsize=7)
    ax.set_xlabel("Fitting window fraction", fontsize=11)
    ax.set_ylabel("Asteroid", fontsize=11)

    # Annotate cells with RMSE km values
    for i in range(len(asteroids)):
        for j in range(len(splits)):
            if not np.isnan(matrix[i, j]):
                val = 10**matrix[i, j]
                txt = f"{val:.2f}" if val < 100 else f"{val:.0f}"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=5.5, color="black" if matrix[i, j] > -1 else "white")

    plt.tight_layout()
    save_fig(fig, "fig7_heatmap_rmse")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  Generating all results figures")
    print("=" * 60)

    print("\nLoading split study data ...")
    df_all = load_all_splits()
    n_ast  = df_all["asteroid"].nunique()
    n_spl  = df_all["split_pct"].nunique()
    print(f"  {n_ast} asteroids × {n_spl} splits = {len(df_all)} rows")

    print("\nLoading MC summary data ...")
    mc_summaries = load_all_mc()
    print(f"  {len(mc_summaries)} asteroids with MC data")

    print("\nFigure 1: RMSE vs split — all asteroids ...")
    fig1_rmse_vs_split(df_all)

    print("Figure 2: RMSE box plots ...")
    fig2_rmse_boxplots(df_all)

    print("Figure 3: A2 stability ...")
    fig3_A2_stability(df_all)

    # Use toutatis as the representative example asteroid.
    # Change this to any asteroid in your dataset.
    EXAMPLE = "toutatis"

    print(f"Figure 4: Split study detail — {EXAMPLE} ...")
    fig4_single_asteroid_split(df_all, EXAMPLE)

    print(f"Figure 5: MC uncertainty — {EXAMPLE} ...")
    fig5_mc_single(mc_summaries, EXAMPLE)

    print("Figure 6: MC comparison — all asteroids ...")
    fig6_mc_comparison_all(mc_summaries)

    print("Figure 7: RMSE heatmap ...")
    fig7_heatmap(df_all)

    print(f"\nAll figures saved to {FIG_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()