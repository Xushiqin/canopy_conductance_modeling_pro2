from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Paths
# =========================
INPUT_FILE = Path("./fluxnet_model_output/DD/gs_summary/DD_combined_gs_summary.csv")

FIG_DIR = Path("./figures")
STAT_DIR = Path("./statistics")

FIG_DIR.mkdir(parents=True, exist_ok=True)
STAT_DIR.mkdir(parents=True, exist_ok=True)

FIG_FILE = FIG_DIR / "DD_gs_kge_box_layer1.png"
STAT_FILE = STAT_DIR / "DD_gs_kge_box_layer1.csv"


# =========================
# Model pairs
# =========================
MODEL_PAIRS = [
    {
        "group": "BBL",
        "left_label": "noSWC",
        "right_label": "mSWC",
        "left_col": "BBL_noSWC_KGE_prime",
        "right_col": "BBL_mSWC_layer_1_KGE_prime",
    },
    {
        "group": "Medlyn",
        "left_label": "noSWC",
        "right_label": "mSWC",
        "left_col": "Medlyn_noSWC_KGE_prime",
        "right_col": "Medlyn_mSWC_layer_1_KGE_prime",
    },
    {
        "group": "RF-mSWC",
        "left_label": "noSWC",
        "right_label": "mSWC",
        "left_col": "RF_GPP_VPD_leaf_noSWC_KGE_prime",
        "right_col": "RF_GPP_VPD_leaf_mSWC_layer_1_KGE_prime",
    },
    {
        "group": "RF-SWC",
        "left_label": "noSWC",
        "right_label": "SWC",
        "left_col": "RF_GPP_VPD_leaf_noSWC_KGE_prime",
        "right_col": "RF_GPP_VPD_leaf_SWC_layer_1_KGE_prime",
    },
]

MODEL_COLUMNS = []
for item in MODEL_PAIRS:
    MODEL_COLUMNS.extend([item["left_col"], item["right_col"]])


# =========================
# Helper functions
# =========================
def load_data(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    return df


def check_columns(df: pd.DataFrame, required_cols: list):
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError("Missing columns:\n" + "\n".join(missing))


def exclude_igbp_classes(df: pd.DataFrame, exclude_classes=None) -> pd.DataFrame:
    if "IGBP" not in df.columns:
        raise KeyError("Missing column: IGBP")

    if exclude_classes is None:
        exclude_classes = ["WET", "CVM"]

    exclude_classes = [str(x).strip().upper() for x in exclude_classes]

    out = df.copy()
    out["IGBP"] = out["IGBP"].astype(str).str.strip().str.upper()

    n_before = len(out)
    excluded_mask = out["IGBP"].isin(exclude_classes)
    n_excluded = excluded_mask.sum()

    for cls in exclude_classes:
        n_cls = (out["IGBP"] == cls).sum()
        print(f"[INFO] Rows excluded (IGBP == {cls}): {n_cls}")

    out = out.loc[~excluded_mask].copy()
    n_after = len(out)

    print(f"[INFO] Rows before excluding target IGBP classes: {n_before}")
    print(f"[INFO] Total rows excluded: {n_excluded}")
    print(f"[INFO] Rows after exclusion: {n_after}")

    return out


def clean_series(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    s = s.where(s >= -0.41, np.nan)
    return s


def get_paired_data(df: pd.DataFrame, col_left: str, col_right: str):
    tmp = pd.DataFrame({
        "left": clean_series(df[col_left]),
        "right": clean_series(df[col_right]),
    })

    n_before = len(tmp)
    tmp = tmp.dropna(subset=["left", "right"]).copy()
    n_after = len(tmp)
    n_removed = n_before - n_after

    print(f"[INFO] Pair filtering: {col_left} vs {col_right}")
    print(f"       rows before pairing = {n_before}")
    print(f"       rows after pairing  = {n_after}")
    print(f"       rows removed        = {n_removed}")

    return tmp["left"], tmp["right"]


def summarize_models(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for item in MODEL_PAIRS:
        s_left, s_right = get_paired_data(df, item["left_col"], item["right_col"])

        rows.append({
            "Group": item["group"],
            "Scenario": item["left_label"],
            "Column": item["left_col"],
            "N": len(s_left),
            "Median_KGE_prime": s_left.median() if len(s_left) > 0 else np.nan,
            "Mean_KGE_prime": s_left.mean() if len(s_left) > 0 else np.nan,
            "Std_KGE_prime": s_left.std() if len(s_left) > 1 else np.nan,
            "Min_KGE_prime": s_left.min() if len(s_left) > 0 else np.nan,
            "Q1_KGE_prime": s_left.quantile(0.25) if len(s_left) > 0 else np.nan,
            "Q3_KGE_prime": s_left.quantile(0.75) if len(s_left) > 0 else np.nan,
            "Max_KGE_prime": s_left.max() if len(s_left) > 0 else np.nan,
        })

        rows.append({
            "Group": item["group"],
            "Scenario": item["right_label"],
            "Column": item["right_col"],
            "N": len(s_right),
            "Median_KGE_prime": s_right.median() if len(s_right) > 0 else np.nan,
            "Mean_KGE_prime": s_right.mean() if len(s_right) > 0 else np.nan,
            "Std_KGE_prime": s_right.std() if len(s_right) > 1 else np.nan,
            "Min_KGE_prime": s_right.min() if len(s_right) > 0 else np.nan,
            "Q1_KGE_prime": s_right.quantile(0.25) if len(s_right) > 0 else np.nan,
            "Q3_KGE_prime": s_right.quantile(0.75) if len(s_right) > 0 else np.nan,
            "Max_KGE_prime": s_right.max() if len(s_right) > 0 else np.nan,
        })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(STAT_FILE, index=False)
    return summary_df


# =========================
# Plot
# =========================
def plot_boxplot_scatter(df: pd.DataFrame):
    _ = summarize_models(df)

    n_panels = len(MODEL_PAIRS)
    fig, axes = plt.subplots(n_panels, 1, figsize=(7, 12), sharex=True)

    if n_panels == 1:
        axes = [axes]

    panel_labels = ["(A)", "(B)", "(C)", "(D)"]

    box_colors = ["#4C72B0", "#55A868"]
    scatter_colors = ["#4C72B0", "#55A868"]

    rng = np.random.default_rng(42)

    for i, item in enumerate(MODEL_PAIRS):
        ax = axes[i]

        s_left, s_right = get_paired_data(df, item["left_col"], item["right_col"])

        data = [s_left.values, s_right.values]
        positions = [1, 2]

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.4),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
            boxprops=dict(linewidth=1.0),
        )

        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.45)

        for j, s in enumerate([s_left, s_right]):
            if len(s) == 0:
                continue

            x_center = positions[j]
            x_jitter = rng.normal(loc=x_center, scale=0.045, size=len(s))

            ax.scatter(
                x_jitter,
                s.values,
                s=16,
                alpha=0.45,
                color=scatter_colors[j],
                edgecolors="none",
                zorder=3,
            )

        for j, s in enumerate([s_left, s_right]):
            if len(s) == 0:
                continue

            med = s.median()
            ax.text(
                positions[j],
                med,
                f"{med:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        n_pair = len(s_left)
        ax.text(
            1.5,
            0.98,
            f"Paired N={n_pair}",
            ha="center",
            va="top",
            fontsize=9,
            transform=ax.get_xaxis_transform(),
        )

        for ref in [0.0, 0.5, 0.75]:
            ax.axhline(ref, linestyle=":", linewidth=0.8, color="gray", zorder=0)

        ax.set_ylabel("KGE'")
        ax.set_title(f"{panel_labels[i]} {item['group']}", loc="left")
        ax.set_xticks([1, 2])
        ax.set_xticklabels([item["left_label"], item["right_label"]])

    axes[-1].set_xlabel("Scenario")

    plt.tight_layout()
    plt.savefig(FIG_FILE, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"[OK] Figure → {FIG_FILE}")
    print(f"[OK] Statistics → {STAT_FILE}")


# =========================
# Main
# =========================
def main():
    df = load_data(INPUT_FILE)
    check_columns(df, MODEL_COLUMNS + ["IGBP"])
    df = exclude_igbp_classes(df, exclude_classes=["WET", "CVM"])
    plot_boxplot_scatter(df)


if __name__ == "__main__":
    main()