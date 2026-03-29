from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# =========================
# Paths
# =========================
INPUT_FILE = Path("./fluxnet/BIF_summary/BIF_model_sites.csv")
OUT_FILE = Path("./figures/global_fluxnet_sites_distribution.jpg")


# =========================
# Load & preprocess data
# =========================
df = pd.read_csv(INPUT_FILE)
df.columns = df.columns.str.strip()

# Convert to numeric
df["LOCATION_LAT"] = pd.to_numeric(df["LOCATION_LAT"], errors="coerce")
df["LOCATION_LONG"] = pd.to_numeric(df["LOCATION_LONG"], errors="coerce")
df["Temporal_extent"] = pd.to_numeric(df["Temporal_extent"], errors="coerce")

# Filter valid sites and latitude range (modified to -60 to 90)
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


# =========================
# Summary statistics
# =========================
total_sites = len(df)
total_years = df["Temporal_extent"].fillna(0).sum()

groups = ["Forests", "Grasslands", "Savannas", "Shrublands"]
colors = {
    "Forests": "#A9D18E",
    "Grasslands": "#FFA3A3",
    "Savannas": "#CDACE6",
    "Shrublands": "#e6b800",
}

summary = []
for g in groups:
    sub = df[df["GROUP"] == g]
    n = len(sub)
    y = sub["Temporal_extent"].fillna(0).sum()
    summary.append((g, n, 100*n/total_sites, y, 100*y/total_years))


# =========================
# Plot map
# =========================
plt.rcParams['font.family'] = 'sans-serif'
fig = plt.figure(figsize=(15, 10))
ax = plt.axes(projection=ccrs.PlateCarree())

ax.set_extent([-180, 180, -60, 90])

# Add ocean fill with light blue and 50% transparency
ax.add_feature(cfeature.OCEAN, facecolor="#ADD8E6", alpha=0.3, edgecolor='none')
# Add land with gray color
ax.add_feature(cfeature.LAND, facecolor="#d9d9d9", edgecolor='none')
# Add coastlines with black line
ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black')

# Scatter with biome-specific colors - larger markers
for g in groups:
    subset = df[df["GROUP"] == g]
    ax.scatter(
        subset["LOCATION_LONG"],
        subset["LOCATION_LAT"],
        s=28,  # Increased from 14 to 28 for larger markers
        facecolor=colors[g],
        edgecolor="black",
        linewidth=0.3,
        transform=ccrs.PlateCarree(),
        zorder=3,
        label=g
    )

# Axis ticks
ax.set_xticks(np.arange(-180, 181, 30))
ax.set_yticks(np.arange(-60, 91, 30))
ax.tick_params(labelsize=14)

# Frame
for s in ax.spines.values():
    s.set_linewidth(1.2)


# =========================
# Legend panel (no fill, no divider line)
# =========================
# Left text - increased distance from map (lower y position)
fig.text(0.12, 0.14, f"Total sites: {int(total_sites):,}", fontsize=14, weight="normal")
fig.text(0.12, 0.09, f"Total site years: {int(total_years):,}", fontsize=14, weight="normal")

# Header - not bold
fig.text(0.36, 0.17, "Biome", fontsize=14, weight="normal")
fig.text(0.60, 0.17, "Site count (%)", fontsize=14, weight="normal")
fig.text(0.78, 0.17, "Site years (%)", fontsize=14, weight="normal")

# Rows - even tighter line spacing
y0 = 0.14
line_spacing = 0.030  # Reduced from 0.035 to 0.030 for tighter spacing
for i, (g, n, p, y, py) in enumerate(summary):
    yy = y0 - i * line_spacing
    fig.text(0.36, yy, "●", color=colors[g], fontsize=14)
    fig.text(0.39, yy, g, fontsize=12)
    # Format percentages to 0 decimal places (whole numbers)
    fig.text(0.60, yy, f"{int(n):,} ({p:.0f}%)", fontsize=12)
    fig.text(0.78, yy, f"{int(y):,} ({py:.0f}%)", fontsize=12)


# =========================
# Save
# =========================
plt.savefig(OUT_FILE, dpi=300, bbox_inches="tight")
plt.close()

print(f"[DONE] Site distribution map saved to: {OUT_FILE}")
print(f"[INFO] Total sites after removing Croplands: {total_sites}")
print(f"[INFO] Total site years after removing Croplands: {int(total_years):,}")