from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================================
# Paths
# =====================================
INPUT_FILE = Path("./fluxnet_model_output/DD/gs_summary/DD_combined_gs_summary.csv")
OUT_DIR = Path("./figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "conceptual_response_gc_et_vpd.jpg"


# =====================================
# Physical constants
# =====================================
RHO_AIR = 1.293       # kg m-3
CP_AIR = 1004.67      # J kg-1 K-1
LAMBDA_V = 2.45e6     # J kg-1


# =====================================
# Time scale
# =====================================
SECONDS = 86400.0   # DD


# =====================================
# Representative environmental parameters
# =====================================
S = 0.20
GAMMA = 0.066
RNET = 200.0
GA = 0.02
GPP = 5.0

VPD = np.linspace(0.1, 4.0, 500)


# =====================================
# Load g1 quantiles
# =====================================
def load_g1_quantiles(csv_file):
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()

    bbl = pd.to_numeric(df["BBL_g1"], errors="coerce")
    med = pd.to_numeric(df["Medlyn_g1"], errors="coerce")

    bbl = bbl[np.isfinite(bbl) & (bbl > 0)]
    med = med[np.isfinite(med) & (med > 0)]

    return np.percentile(bbl, [25, 50, 75]), np.percentile(med, [25, 50, 75])


# =====================================
# Model functions
# =====================================
def gc_bbl(vpd, g1):
    return g1 * GPP / vpd


def gc_medlyn(vpd, g1):
    return g1 * GPP / np.sqrt(vpd)


def dET_bbl(vpd, g1):
    num = RHO_AIR * CP_AIR * (S + GAMMA) - (GAMMA * S * RNET) / (g1 * GPP)
    den = (S + GAMMA + (GAMMA * GA / (g1 * GPP)) * vpd) ** 2
    return (SECONDS / LAMBDA_V) * GA * num / den


def dET_medlyn(vpd, g1):
    num = (
        RHO_AIR * CP_AIR * (S + GAMMA)
        + (RHO_AIR * CP_AIR * GAMMA * GA / (2 * g1 * GPP)) * np.sqrt(vpd)
        - (GAMMA * S * RNET) / (2 * g1 * GPP * np.sqrt(vpd))
    )
    den = (S + GAMMA + (GAMMA * GA / (g1 * GPP)) * np.sqrt(vpd)) ** 2
    return (SECONDS / LAMBDA_V) * GA * num / den


# =====================================
# Styling
# =====================================
def add_label(ax, label):
    ax.text(0, 1.05, label, transform=ax.transAxes,
            fontsize=14, ha="left", va="bottom")


def style(ax):
    ax.grid(False)
    for s in ax.spines.values():
        s.set_linewidth(1)


# =====================================
# Main
# =====================================
def main():
    bbl_q, med_q = load_g1_quantiles(INPUT_FILE)

    bbl_p25, bbl_p50, bbl_p75 = bbl_q
    med_p25, med_p50, med_p75 = med_q

    # 🔥 Final color scheme (your request)
    colors = {
        "P25": "#fdae61",   # orange (dry)
        "P50": "#1a9850",   # green (mid)
        "P75": "#4575b4"    # blue (wet)
    }

    lw = {"P25": 2.0, "P50": 2.6, "P75": 2.0}

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.8))

    axA = axes[0, 0]
    axB = axes[1, 0]
    axC = axes[0, 1]
    axD = axes[1, 1]

    # ---------------- A (BBL gc)
    axA.plot(VPD, gc_bbl(VPD, bbl_p25), color=colors["P25"], lw=lw["P25"],
             label="25% $g_1$ (Dry)")
    axA.plot(VPD, gc_bbl(VPD, bbl_p50), color=colors["P50"], lw=lw["P50"],
             label="Median $g_1$ (Moderate)")
    axA.plot(VPD, gc_bbl(VPD, bbl_p75), color=colors["P75"], lw=lw["P75"],
             label="75% $g_1$ (Wet)")

    axA.set_ylabel(r"$g_c$ (m s$^{-1}$)")
    axA.legend(frameon=False)
    add_label(axA, "A")
    style(axA)

    # ---------------- C (BBL dET/dVPD)
    axB.plot(VPD, dET_bbl(VPD, bbl_p25), color=colors["P25"], lw=lw["P25"])
    axB.plot(VPD, dET_bbl(VPD, bbl_p50), color=colors["P50"], lw=lw["P50"])
    axB.plot(VPD, dET_bbl(VPD, bbl_p75), color=colors["P75"], lw=lw["P75"])

    axB.axhline(0, color="black", lw=1.0, linestyle="--")
    axB.set_xlabel("VPD (kPa)")
    axB.set_ylabel(r"$\partial ET / \partial VPD$ (mm day$^{-1}$ kPa$^{-1}$)")
    add_label(axB, "C")
    style(axB)

    # ---------------- B (Medlyn gc)
    axC.plot(VPD, gc_medlyn(VPD, med_p25), color=colors["P25"], lw=lw["P25"])
    axC.plot(VPD, gc_medlyn(VPD, med_p50), color=colors["P50"], lw=lw["P50"])
    axC.plot(VPD, gc_medlyn(VPD, med_p75), color=colors["P75"], lw=lw["P75"])

    axC.set_ylabel(r"$g_c$ (m s$^{-1}$)")
    add_label(axC, "B")
    style(axC)

    # ---------------- D (Medlyn dET/dVPD)
    axD.plot(VPD, dET_medlyn(VPD, med_p25), color=colors["P25"], lw=lw["P25"])
    axD.plot(VPD, dET_medlyn(VPD, med_p50), color=colors["P50"], lw=lw["P50"])
    axD.plot(VPD, dET_medlyn(VPD, med_p75), color=colors["P75"], lw=lw["P75"])

    axD.axhline(0, color="black", lw=1.0, linestyle="--")
    axD.set_xlabel("VPD (kPa)")
    axD.set_ylabel(r"$\partial ET / \partial VPD$ (mm day$^{-1}$ kPa$^{-1}$)")
    add_label(axD, "D")
    style(axD)

    # Column titles
    axA.set_title("BBL", fontsize=14, pad = 40)
    axC.set_title("Medlyn", fontsize=14, pad = 40)
    

    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved to: {OUT_FILE}")


if __name__ == "__main__":
    main()