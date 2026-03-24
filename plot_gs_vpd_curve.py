import numpy as np
import matplotlib.pyplot as plt


# =========================
# Parameters
# =========================
g0 = 0.0
g1 = 2.0
m_sm = 0.5   # example soil moisture constraint

# =========================
# VPD range
# =========================
vpd = np.linspace(0.05, 3.0, 300)

# =========================
# No SWC
# =========================
gs_bbl = g0 + g1 / vpd
gs_medlyn = g0 + g1 / np.sqrt(vpd)

# =========================
# With SWC constraint
# =========================
gs_bbl_swc = gs_bbl * m_sm
gs_medlyn_swc = gs_medlyn * m_sm

# =========================
# Plot
# =========================
plt.figure(figsize=(7, 5))

plt.plot(vpd, gs_bbl, linewidth=2, label="BBL (no SWC)")
plt.plot(vpd, gs_bbl_swc, linewidth=2, linestyle="--", label="BBL × mSM")
plt.plot(vpd, gs_medlyn, linewidth=2, label="Medlyn (no SWC)")
plt.plot(vpd, gs_medlyn_swc, linewidth=2, linestyle="--", label="Medlyn × mSM")

plt.xlabel("VPD (kPa)", fontsize=12)
plt.ylabel("gs (relative)", fontsize=12)

plt.xlim(0, 3)
plt.ylim(0, max(gs_bbl) * 1.05)

plt.text(
    1.45, max(gs_bbl) * 0.82,
    r"BBL: $g_s=g_0+g_1\frac{1}{VPD}$",
    fontsize=10
)
plt.text(
    1.45, max(gs_bbl) * 0.72,
    r"BBL with SWC: $g_s=\left(g_0+g_1\frac{1}{VPD}\right)m_{SM}$",
    fontsize=10
)
plt.text(
    1.45, max(gs_bbl) * 0.58,
    r"Medlyn: $g_s=g_0+g_1\frac{1}{\sqrt{VPD}}$",
    fontsize=10
)
plt.text(
    1.45, max(gs_bbl) * 0.48,
    r"Medlyn with SWC: $g_s=\left(g_0+g_1\frac{1}{\sqrt{VPD}}\right)m_{SM}$",
    fontsize=10
)

plt.legend(frameon=False)
plt.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("./figures/VPD_gs_theoretical_curves_with_SWC.png", dpi=300)
plt.show()