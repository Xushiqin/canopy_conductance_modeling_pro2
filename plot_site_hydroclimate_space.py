from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================
# Paths
# =========================
INPUT_FILE = Path("./fluxnet/BIF_summary/BIF_model_sites.csv")
OUT_FILE = Path("./figures/dryness_vs_evaporation_index.png")


# =========================
# Load & preprocess data
# =========================
df = pd.read_csv(INPUT_FILE)
df.columns = df.columns.str.strip()

# Convert to numeric
df["dryness_index_mean"] = pd.to_numeric(df["dryness_index_mean"], errors="coerce")
df["evaporation_index_mean"] = pd.to_numeric(df["evaporation_index_mean"], errors="coerce")
df["LOCATION_LAT"] = pd.to_numeric(df["LOCATION_LAT"], errors="coerce")
df["LOCATION_LONG"] = pd.to_numeric(df["LOCATION_LONG"], errors="coerce")
df["Temporal_extent"] = pd.to_numeric(df["Temporal_extent"], errors="coerce")

# Filter valid sites and latitude range
df = df.dropna(subset=["LOCATION_LAT", "LOCATION_LONG"])
df = df[(df["LOCATION_LAT"] >= -60) & (df["LOCATION_LAT"] <= 90)]


# =========================
# IGBP regroup (WSA → Savannas, exclude Croplands)
# =========================
def igbp_group(x):
    x = str(x).strip().upper()
    if x in {"ENF", "EBF", "DBF", "DNF", "MF"}:
        return "Forests"
    elif x == "GRA":
        return "Grasslands"
    elif x in {"SAV", "WSA"}:
        return "Savannas"
    elif x in {"OSH", "CSH"}:
        return "Shrublands"
    elif x == "CRO":
        return None  # Exclude Croplands
    else:
        return None

df["GROUP"] = df["IGBP"].apply(igbp_group)

# Remove rows with None group (including Croplands)
df = df.dropna(subset=["GROUP"]).copy()

# Remove rows with NaN in the indices for scatter plot
scatter_df = df.dropna(subset=["dryness_index_mean", "evaporation_index_mean"]).copy()


# =========================
# Groups and colors
# =========================
groups = ["Forests", "Grasslands", "Savannas", "Shrublands"]
colors = {
    "Forests": "#A9D18E",
    "Grasslands": "#FFA3A3",
    "Savannas": "#CDACE6",
    "Shrublands": "#e6b800",
}


# =========================
# Create scatter plot with transparent background
# =========================
plt.rcParams['font.family'] = 'sans-serif'
fig, ax = plt.subplots(figsize=(6, 4))

# Set transparent background for figure and axes
fig.patch.set_alpha(0)  # Make figure background transparent
ax.patch.set_alpha(0)   # Make axes background transparent

# Plot points with biome-specific colors
for g in groups:
    subset = scatter_df[scatter_df["GROUP"] == g]
    ax.scatter(
        subset["dryness_index_mean"],
        subset["evaporation_index_mean"],
        s=100,
        facecolor=colors[g],
        edgecolor="black",
        linewidth=1,
        alpha=0.7,
        zorder=2
    )

# Set labels with italic formatting
ax.set_xlabel(r"Dryness index ($PET/P$)", fontsize=18)
ax.set_ylabel(r"Evaporation index ($ET/P$)", fontsize=18)

# Set axis limits with some padding
x_min = scatter_df["dryness_index_mean"].min()
x_max = scatter_df["dryness_index_mean"].max()
y_min = scatter_df["evaporation_index_mean"].min()
y_max = scatter_df["evaporation_index_mean"].max()

x_pad = (x_max - x_min) * 0.05
y_pad = (y_max - y_min) * 0.05

ax.set_xlim(x_min - x_pad, x_max + x_pad)
ax.set_ylim(y_min - y_pad, y_max + y_pad)

# Set tick label size
ax.tick_params(labelsize=16)

# Make the plot tight
plt.tight_layout()

# Save with transparent background
plt.savefig(OUT_FILE, dpi=300, bbox_inches="tight", transparent=True)
plt.close()

print(f"[DONE] Scatter plot saved to: {OUT_FILE}")
print(f"[INFO] Number of points plotted: {len(scatter_df)}")
print(f"[INFO] Dryness index range: {x_min:.2f} - {x_max:.2f}")
print(f"[INFO] Evaporation index range: {y_min:.2f} - {y_max:.2f}")