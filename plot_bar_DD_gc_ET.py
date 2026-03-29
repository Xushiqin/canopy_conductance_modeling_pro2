from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# =========================
# Input files
# =========================
GS_INPUT_FILE = Path("./fluxnet_model_output/DD/gs_summary/DD_combined_gs_summary.csv")
ET_INPUT_FILE = Path("./fluxnet_model_output/DD/ET_summary/DD_combined_ET_summary.csv")

FIG_DIR = Path("./figures")
STAT_DIR = Path("./statistics")

FIG_DIR.mkdir(parents=True, exist_ok=True)
STAT_DIR.mkdir(parents=True, exist_ok=True)

FIG_FILE = FIG_DIR / "DD_gc_ET_barplot.jpg"
STAT_FILE = STAT_DIR / "DD_gc_ET_barplot.csv"


# =========================
# Plot design
# =========================
MODEL_PAIRS = [
    {
        "group": "BBL",
        "left_label_math": r"$\mathrm{BBL}_{\mathrm{baseline}}$",
        "right_label_math": r"$\mathrm{BBL}_{\beta_{\mathrm{JS}}}$",
        "left_label_plain": "BBL_baseline",
        "right_label_plain": "BBL_beta_JS",
        "left_prefix": "BBL_noSWC",
        "right_prefix": "BBL_mSWC_layer_1",
    },
    {
        "group": "Medlyn",
        "left_label_math": r"$\mathrm{Medlyn}_{\mathrm{baseline}}$",
        "right_label_math": r"$\mathrm{Medlyn}_{\beta_{\mathrm{JS}}}$",
        "left_label_plain": "Medlyn_baseline",
        "right_label_plain": "Medlyn_beta_JS",
        "left_prefix": "Medlyn_noSWC",
        "right_prefix": "Medlyn_mSWC_layer_1",
    },
    {
        "group": "RF_JS",
        "left_label_math": r"$\mathrm{RF}_{\mathrm{baseline}}$",
        "right_label_math": r"$\mathrm{RF}_{\beta_{\mathrm{JS}}}$",
        "left_label_plain": "RF_baseline",
        "right_label_plain": "RF_beta_JS",
        "left_prefix": "RF_GPP_VPD_leaf_noSWC",
        "right_prefix": "RF_GPP_VPD_leaf_mSWC_layer_1",
    },
    {
        "group": "RF_RF",
        "left_label_math": r"$\mathrm{RF}_{\mathrm{baseline}}$",
        "right_label_math": r"$\mathrm{RF}_{\beta_{\mathrm{RF}}}$",
        "left_label_plain": "RF_baseline",
        "right_label_plain": "RF_beta_RF",
        "left_prefix": "RF_GPP_VPD_leaf_noSWC",
        "right_prefix": "RF_GPP_VPD_leaf_SWC_layer_1",
    },
]

METRICS = [
    {"row_name": "KGE'", "suffix": "KGE_prime", "color": "#FFA3A3"},
    {"row_name": "Correlation", "suffix": "r", "color": "#A9D18E"},
    {"row_name": "Variability", "suffix": "gamma", "color": "#CDACE6"},
    {"row_name": "Bias", "suffix": "beta", "color": "#79ADDD"},
]


# =========================
# Helpers
# =========================
def load_data(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    return df


def exclude_igbp_classes(df: pd.DataFrame, exclude_classes=None) -> pd.DataFrame:
    if exclude_classes is None:
        exclude_classes = ["CRO", "WET", "CVM"]

    out = df.copy()
    if "IGBP" not in out.columns:
        raise KeyError("Missing column: IGBP")

    out["IGBP"] = out["IGBP"].astype(str).str.strip().str.upper()
    exclude_classes = [str(x).strip().upper() for x in exclude_classes]

    before = len(out)
    out = out.loc[~out["IGBP"].isin(exclude_classes)].copy()
    after = len(out)

    print(f"[INFO] Rows before IGBP exclusion: {before}")
    print(f"[INFO] Rows after IGBP exclusion : {after}")
    print(f"[INFO] Rows excluded             : {before - after}")

    return out


def get_metric_column(prefix: str, suffix: str) -> str:
    return f"{prefix}_{suffix}"


def check_required_columns(df: pd.DataFrame, dataset_name: str):
    required = ["IGBP"]
    for pair in MODEL_PAIRS:
        for metric in METRICS:
            required.append(get_metric_column(pair["left_prefix"], metric["suffix"]))
            required.append(get_metric_column(pair["right_prefix"], metric["suffix"]))

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing columns in {dataset_name}:\n" + "\n".join(missing)
        )


def clean_metric_series(s: pd.Series, metric_suffix: str) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)

    if metric_suffix == "KGE_prime":
        s = s.where(s >= -0.41, np.nan)

    return s


def get_paired_values(df: pd.DataFrame, left_col: str, right_col: str, metric_suffix: str):
    tmp = pd.DataFrame({
        "left": clean_metric_series(df[left_col], metric_suffix),
        "right": clean_metric_series(df[right_col], metric_suffix),
    })
    tmp = tmp.dropna(subset=["left", "right"]).copy()
    return tmp["left"], tmp["right"]


def summarize_dataset(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    rows = []

    for pair in MODEL_PAIRS:
        for metric in METRICS:
            left_col = get_metric_column(pair["left_prefix"], metric["suffix"])
            right_col = get_metric_column(pair["right_prefix"], metric["suffix"])

            left_vals, right_vals = get_paired_values(df, left_col, right_col, metric["suffix"])

            rows.append({
                "Dataset": dataset_name,
                "Group": pair["group"],
                "Metric": metric["row_name"],
                "Scenario": pair["left_label_plain"],
                "Column": left_col,
                "N": len(left_vals),
                "Median": left_vals.median() if len(left_vals) else np.nan,
                "SD": left_vals.std(ddof=1) if len(left_vals) > 1 else np.nan,
            })

            rows.append({
                "Dataset": dataset_name,
                "Group": pair["group"],
                "Metric": metric["row_name"],
                "Scenario": pair["right_label_plain"],
                "Column": right_col,
                "N": len(right_vals),
                "Median": right_vals.median() if len(right_vals) else np.nan,
                "SD": right_vals.std(ddof=1) if len(right_vals) > 1 else np.nan,
            })

    return pd.DataFrame(rows)


def get_metric_ylim(summary_df: pd.DataFrame, metric_name: str):
    sub = summary_df.loc[summary_df["Metric"] == metric_name].copy()
    vals = sub["Median"].to_numpy(dtype=float)

    low = np.nanmin(vals)
    high = np.nanmax(vals)

    if not np.isfinite(low):
        low = 0.0
    if not np.isfinite(high):
        high = 1.0

    pad = 0.14 * (high - low) if high > low else 0.1
    low -= pad
    high += pad

    if metric_name in ["KGE'", "Correlation"]:
        low = max(low, -0.5)
        high = min(high, 1.05)

    return low, high


def add_n_labels(ax, bars, n_values, fontsize=10):
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    offset = 0.02 * y_range

    for bar, n in zip(bars, n_values):
        height = bar.get_height()
        if pd.isna(height):
            continue

        x = bar.get_x() + bar.get_width() / 2
        y = height + offset

        ax.text(
            x, y, f"n={int(n)}",
            ha="center", va="bottom",
            fontsize=fontsize
        )


def get_panel_median(heights):
    vals = pd.to_numeric(pd.Series(heights), errors="coerce").dropna()
    if len(vals) == 0:
        return np.nan
    return float(vals.median())


def plot_figure(gs_df: pd.DataFrame, et_df: pd.DataFrame):
    gs_summary = summarize_dataset(gs_df, "gc")
    et_summary = summarize_dataset(et_df, "ET")
    summary_df = pd.concat([gs_summary, et_summary], ignore_index=True)
    summary_df.to_csv(STAT_FILE, index=False)

    fig, axes = plt.subplots(
        nrows=4,
        ncols=2,
        figsize=(14, 18)
    )

    # Reduce row spacing to half of previous 0.42
    fig.subplots_adjust(hspace=0.21, wspace=0.22)

    panel_labels = ["A", "B", "C", "D", "E", "F", "G", "H"]

    x = np.arange(8)
    bar_width = 0.72

    summaries = {"gc": gs_summary, "ET": et_summary}

    for row_idx, metric in enumerate(METRICS):
        metric_name = metric["row_name"]
        metric_color = metric["color"]
        y_low, y_high = get_metric_ylim(summary_df, metric_name)

        for col_idx, dataset_name in enumerate(["gc", "ET"]):
            ax = axes[row_idx, col_idx]
            dataset_summary = summaries[dataset_name]

            heights = []
            n_values = []
            xlabels = []
            edgecolors = []
            hatch_styles = []
            facecolors = []

            for pair in MODEL_PAIRS:
                sub = dataset_summary[
                    (dataset_summary["Metric"] == metric_name) &
                    (dataset_summary["Group"] == pair["group"])
                ].copy()

                left_row = sub[sub["Scenario"] == pair["left_label_plain"]].iloc[0]
                right_row = sub[sub["Scenario"] == pair["right_label_plain"]].iloc[0]

                heights.extend([left_row["Median"], right_row["Median"]])
                n_values.extend([left_row["N"], right_row["N"]])
                xlabels.extend([pair["left_label_math"], pair["right_label_math"]])

                # Original metric color only for edge color
                edgecolors.extend([metric_color, metric_color])

                # Baseline: no fill
                facecolors.append("white")
                hatch_styles.append("")

                # SWC constrained: dotted fill
                facecolors.append("white")
                hatch_styles.append(".")

            bars = ax.bar(
                x,
                heights,
                width=bar_width,
                color=facecolors,
                edgecolor=edgecolors,
                linewidth=1.4,
                zorder=3,
            )

            for bar, hatch in zip(bars, hatch_styles):
                bar.set_hatch(hatch)

            ax.set_ylim(y_low, y_high)
            ax.set_axisbelow(True)
            ax.grid(False)

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.0)

            ax.set_ylabel(metric_name, fontsize=13)

            ax.set_xticks(x)

            if row_idx == 3:
                ax.set_xticklabels(
                    xlabels,
                    rotation=90,
                    ha="center",
                    va="top",
                    fontsize=12
                )
            else:
                ax.set_xticklabels([""] * len(xlabels))
                ax.tick_params(axis="x", length=3)

            ax.tick_params(axis="x", labelsize=12)
            ax.tick_params(axis="y", labelsize=12)

            if row_idx == 0:
                if dataset_name == "gc":
                    ax.set_title(r"$g_c$", fontsize=16, pad=32)
                else:
                    ax.set_title("$ET$", fontsize=16, pad=32)

            # Place panel label outside the top-left corner
            ax.text(
                0, 1.05,
                panel_labels[row_idx * 2 + col_idx],
                transform=ax.transAxes,
                ha="left", va="bottom",
                fontsize=13,
                fontweight="normal"
            )

            add_n_labels(ax, bars, n_values, fontsize=10)

            # Only one horizontal line: panel median, in metric color
            panel_median = get_panel_median(heights)
            if np.isfinite(panel_median):
                ax.axhline(
                    panel_median,
                    linestyle="-.",
                    linewidth=1.2,
                    color=metric_color,
                    zorder=1
                )

            if row_idx == 0 and col_idx == 0:
                legend_handles = [
                    Patch(
                        facecolor="white",
                        edgecolor=metric_color,
                        hatch="",
                        label="Baseline",
                        linewidth=1.4
                    ),
                    Patch(
                        facecolor="white",
                        edgecolor=metric_color,
                        hatch=".",
                        label="SWC constrained",
                        linewidth=1.4
                    ),
                ]
                ax.legend(
                    handles=legend_handles,
                    loc="upper right",
                    ncol=2,
                    frameon=False,
                    fontsize=12,
                    borderaxespad=0.3,
                    handlelength=1.8,
                    columnspacing=1.2,
                    handletextpad=0.6
                )

    plt.savefig(FIG_FILE, dpi=300, bbox_inches="tight", format="jpg")
    plt.show()

    print(f"[OK] Figure saved to: {FIG_FILE}")
    print(f"[OK] Statistics saved to: {STAT_FILE}")


def main():
    gs_df = load_data(GS_INPUT_FILE)
    et_df = load_data(ET_INPUT_FILE)

    check_required_columns(gs_df, "gc")
    check_required_columns(et_df, "ET")

    gs_df = exclude_igbp_classes(gs_df, exclude_classes=["CRO", "WET", "CVM"])
    et_df = exclude_igbp_classes(et_df, exclude_classes=["CRO", "WET", "CVM"])

    plot_figure(gs_df, et_df)


if __name__ == "__main__":
    main()