from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib.colors import to_rgba


# =========================
# Global style
# =========================
plt.rcParams["font.family"] = "Sans Serif"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"


# =========================
# Paths
# =========================
gs_INPUT_FILE = Path("./fluxnet_model_output/DD/gs_summary/DD_combined_gs_summary.csv")
ET_INPUT_FILE = Path("./fluxnet_model_output/DD/ET_summary/DD_combined_ET_summary.csv")

FIG_DIR = Path("./figures")
STAT_DIR = Path("./statistics")

FIG_DIR.mkdir(parents=True, exist_ok=True)
STAT_DIR.mkdir(parents=True, exist_ok=True)

FIG_FILE = FIG_DIR / "DD_deltaKGE_vs_dryness_index_biome_gc_ET_2col_4row.jpg"

gs_STAT_FILE = STAT_DIR / "DD_gs_deltaKGE_vs_dryness_index_summary.csv"
ET_STAT_FILE = STAT_DIR / "DD_ET_deltaKGE_vs_dryness_index_summary.csv"

gs_PAIRED_FILE = STAT_DIR / "DD_gs_deltaKGE_paired_valid_sites.csv"
ET_PAIRED_FILE = STAT_DIR / "DD_ET_deltaKGE_paired_valid_sites.csv"

gs_BIOME_REG_STAT_FILE = STAT_DIR / "DD_gs_deltaKGE_vs_dryness_index_biome_linear_CI_regression_stats.csv"
ET_BIOME_REG_STAT_FILE = STAT_DIR / "DD_ET_deltaKGE_vs_dryness_index_biome_linear_CI_regression_stats.csv"

gs_GLOBAL_REG_STAT_FILE = STAT_DIR / "DD_gs_deltaKGE_vs_dryness_index_global_linear_CI_regression_stats.csv"
ET_GLOBAL_REG_STAT_FILE = STAT_DIR / "DD_ET_deltaKGE_vs_dryness_index_global_linear_CI_regression_stats.csv"


# =========================
# User settings
# =========================
EXCLUDE_IGBP = ["WET", "CVM", "CRO"]
POINT_SIZE = 36
POINT_ALPHA = 0.90
KGE_THRESHOLD = -0.41

X_COL = "dryness_index_mean"
IGBP_COL = "IGBP"


# =========================
# Configs
# Panel labels are ordered left -> right:
# Row1: A, B
# Row2: C, D
# Row3: E, F
# Row4: G, H
# =========================
MODEL_CONFIGS_gs = [
    {
        "panel": "A",
        "name": "BBL",
        "ycol_new": "deltaKGE_BBL_beta_JS",
        "baseline_col": "BBL_noSWC_KGE_prime",
        "improved_col": "BBL_mSWC_layer_1_KGE_prime",
        "ylabel": r"$\Delta$KGE$'$",
    },
    {
        "panel": "C",
        "name": "Medlyn",
        "ycol_new": "deltaKGE_Medlyn_beta_JS",
        "baseline_col": "Medlyn_noSWC_KGE_prime",
        "improved_col": "Medlyn_mSWC_layer_1_KGE_prime",
        "ylabel": r"$\Delta$KGE$'$",
    },
    {
        "panel": "E",
        "name": r"RF$_{\beta_{\mathrm{JS}}}$",
        "ycol_new": "deltaKGE_RF_beta_JS",
        "baseline_col": "RF_GPP_VPD_leaf_noSWC_KGE_prime",
        "improved_col": "RF_GPP_VPD_leaf_mSWC_layer_1_KGE_prime",
        "ylabel": r"$\Delta$KGE$'$",
    },
    {
        "panel": "G",
        "name": r"RF$_{\beta_{\mathrm{RF}}}$",
        "ycol_new": "deltaKGE_RF_beta_RF",
        "baseline_col": "RF_GPP_VPD_leaf_noSWC_KGE_prime",
        "improved_col": "RF_GPP_VPD_leaf_SWC_layer_1_KGE_prime",
        "ylabel": r"$\Delta$KGE$'$",
    },
]

MODEL_CONFIGS_ET = [
    {
        "panel": "B",
        "name": "BBL",
        "ycol_new": "deltaKGE_BBL_beta_JS",
        "baseline_col": "BBL_noSWC_KGE_prime",
        "improved_col": "BBL_mSWC_layer_1_KGE_prime",
        "ylabel": r"$\Delta$KGE$'$",
    },
    {
        "panel": "D",
        "name": "Medlyn",
        "ycol_new": "deltaKGE_Medlyn_beta_JS",
        "baseline_col": "Medlyn_noSWC_KGE_prime",
        "improved_col": "Medlyn_mSWC_layer_1_KGE_prime",
        "ylabel": r"$\Delta$KGE$'$",
    },
    {
        "panel": "F",
        "name": r"RF$_{\beta_{\mathrm{JS}}}$",
        "ycol_new": "deltaKGE_RF_beta_JS",
        "baseline_col": "RF_GPP_VPD_leaf_noSWC_KGE_prime",
        "improved_col": "RF_GPP_VPD_leaf_mSWC_layer_1_KGE_prime",
        "ylabel": r"$\Delta$KGE$'$",
    },
    {
        "panel": "H",
        "name": r"RF$_{\beta_{\mathrm{RF}}}$",
        "ycol_new": "deltaKGE_RF_beta_RF",
        "baseline_col": "RF_GPP_VPD_leaf_noSWC_KGE_prime",
        "improved_col": "RF_GPP_VPD_leaf_SWC_layer_1_KGE_prime",
        "ylabel": r"$\Delta$KGE$'$",
    },
]


# =========================
# IGBP -> biome grouping
# =========================
IGBP_GROUP_LABELS = {
    "ENF": "Forests",
    "EBF": "Forests",
    "DNF": "Forests",
    "DBF": "Forests",
    "MF": "Forests",
    "GRA": "Grasslands",
    "SAV": "Savannas",
    "WSA": "Savannas",
    "OSH": "Shrublands",
    "CSH": "Shrublands",
    "WET": "Wetlands",
    "CRO": "Croplands",
}

GROUP_ORDER = ["Forests", "Grasslands", "Savannas", "Shrublands"]

GROUP_COLOR_MAP = {
    "Forests": "#A9D18E",
    "Grasslands": "#FFA3A3",
    "Savannas": "#CDACE6",
    "Shrublands": "#e6b800",
}


# =========================
# Helpers
# =========================
def load_data(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    return df


def standardize_igbp(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if IGBP_COL not in out.columns:
        raise KeyError(f"Missing column: {IGBP_COL}")
    out[IGBP_COL] = out[IGBP_COL].astype(str).str.strip().str.upper()
    out[IGBP_COL] = out[IGBP_COL].replace({"NAN": "Unknown", "": "Unknown"})
    return out


def exclude_igbp_classes(df: pd.DataFrame, exclude_classes=None) -> pd.DataFrame:
    if exclude_classes is None:
        return df.copy()

    out = df.copy()
    exclude_classes = [str(x).strip().upper() for x in exclude_classes]

    before = len(out)
    out = out.loc[~out[IGBP_COL].isin(exclude_classes)].copy()
    after = len(out)

    print(f"[INFO] Rows before exclusion: {before}")
    print(f"[INFO] Rows after exclusion : {after}")
    print(f"[INFO] Rows excluded        : {before - after}")
    print(f"[INFO] Excluded classes     : {exclude_classes}")

    return out


def check_required_columns(df: pd.DataFrame, model_configs: list):
    required = [X_COL, IGBP_COL]
    for cfg in model_configs:
        required.extend([cfg["baseline_col"], cfg["improved_col"]])

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError("Missing required columns:\n" + "\n".join(missing))


def clean_metric_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    return s


def guess_component_col(df: pd.DataFrame, metric_col: str, component: str):
    candidates = []

    if metric_col.endswith("_KGE_prime"):
        prefix = metric_col[:-10]
        candidates.extend([
            f"{prefix}_{component}",
            f"{prefix}_{component}_prime",
            f"{prefix}_{component.upper()}",
            f"{prefix}_{component.upper()}_prime",
        ])

    candidates.extend([
        metric_col.replace("_KGE_prime", f"_{component}"),
        metric_col.replace("_KGE_prime", f"_{component}_prime"),
        metric_col.replace("KGE_prime", component),
        metric_col.replace("KGE_prime", f"{component}_prime"),
    ])

    for c in candidates:
        if c in df.columns:
            return c
    return None


def add_delta_columns(df: pd.DataFrame, model_configs: list) -> pd.DataFrame:
    out = df.copy()

    for cfg in model_configs:
        baseline = clean_metric_series(out[cfg["baseline_col"]])
        improved = clean_metric_series(out[cfg["improved_col"]])

        valid_pair = (
            baseline.notna()
            & improved.notna()
            & (baseline >= KGE_THRESHOLD)
            & (improved >= KGE_THRESHOLD)
        )

        out[cfg["ycol_new"]] = np.nan
        out.loc[valid_pair, cfg["ycol_new"]] = improved.loc[valid_pair] - baseline.loc[valid_pair]

        print(f"[INFO] {cfg['name']}: paired valid sites for {cfg['ycol_new']} = {int(valid_pair.sum())}")

    return out


def get_group_label(igbp: str) -> str:
    igbp = str(igbp).strip().upper()
    return IGBP_GROUP_LABELS.get(igbp, "Unknown")


def prepare_panel_data(df: pd.DataFrame, ycol: str) -> pd.DataFrame:
    tmp = df[[X_COL, IGBP_COL, ycol]].copy()
    tmp[X_COL] = pd.to_numeric(tmp[X_COL], errors="coerce")
    tmp[ycol] = pd.to_numeric(tmp[ycol], errors="coerce")
    tmp = tmp.replace([np.inf, -np.inf], np.nan)
    tmp = tmp.dropna(subset=[X_COL, ycol]).copy()
    tmp["group_label"] = tmp[IGBP_COL].apply(get_group_label)
    tmp = tmp.loc[tmp["group_label"].isin(GROUP_ORDER)].copy()
    return tmp


def fit_linear_regression(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    X = pd.DataFrame({"const": 1.0, "x": x})
    model = sm.OLS(y, X).fit()

    x_grid = np.linspace(np.nanmin(x), np.nanmax(x), 200)
    X_pred = pd.DataFrame({"const": 1.0, "x": x_grid})
    pred = model.get_prediction(X_pred).summary_frame(alpha=0.05)

    return {
        "model": model,
        "x_grid": x_grid,
        "y_fit": pred["mean"].to_numpy(),
        "ci_low": pred["mean_ci_lower"].to_numpy(),
        "ci_high": pred["mean_ci_upper"].to_numpy(),
        "r2": model.rsquared,
        "p": model.f_pvalue,
        "slope": model.params["x"] if "x" in model.params.index else np.nan,
        "intercept": model.params["const"] if "const" in model.params.index else np.nan,
    }


def collect_statistics(df: pd.DataFrame, model_configs: list) -> pd.DataFrame:
    rows = []

    for cfg in model_configs:
        panel_df = prepare_panel_data(df, cfg["ycol_new"])
        n = len(panel_df)

        rows.append({
            "Model": cfg["name"],
            "Y_column": cfg["ycol_new"],
            "N": n,
            "x_min": panel_df[X_COL].min() if n > 0 else np.nan,
            "x_max": panel_df[X_COL].max() if n > 0 else np.nan,
            "y_min": panel_df[cfg["ycol_new"]].min() if n > 0 else np.nan,
            "y_max": panel_df[cfg["ycol_new"]].max() if n > 0 else np.nan,
            "y_mean": panel_df[cfg["ycol_new"]].mean() if n > 0 else np.nan,
            "y_median": panel_df[cfg["ycol_new"]].median() if n > 0 else np.nan,
        })

    return pd.DataFrame(rows)


def collect_biome_regression_statistics(df: pd.DataFrame, model_configs: list) -> pd.DataFrame:
    rows = []

    for cfg in model_configs:
        panel_df = prepare_panel_data(df, cfg["ycol_new"])

        for biome in GROUP_ORDER:
            sub = panel_df.loc[panel_df["group_label"] == biome].copy()
            n = len(sub)

            row = {
                "Model": cfg["name"],
                "Biome": biome,
                "N": n,
                "slope": np.nan,
                "intercept": np.nan,
                "R2": np.nan,
                "p_value": np.nan,
                "x_min": sub[X_COL].min() if n > 0 else np.nan,
                "x_max": sub[X_COL].max() if n > 0 else np.nan,
            }

            if n > 1:
                try:
                    fit = fit_linear_regression(sub[X_COL].values, sub[cfg["ycol_new"]].values)
                    row["slope"] = fit["slope"]
                    row["intercept"] = fit["intercept"]
                    row["R2"] = fit["r2"]
                    row["p_value"] = fit["p"]
                except Exception:
                    pass

            rows.append(row)

    return pd.DataFrame(rows)


def collect_global_regression_statistics(df: pd.DataFrame, model_configs: list) -> pd.DataFrame:
    rows = []

    for cfg in model_configs:
        panel_df = prepare_panel_data(df, cfg["ycol_new"])
        n = len(panel_df)

        row = {
            "Model": cfg["name"],
            "N": n,
            "slope": np.nan,
            "intercept": np.nan,
            "R2": np.nan,
            "p_value": np.nan,
            "x_min": panel_df[X_COL].min() if n > 0 else np.nan,
            "x_max": panel_df[X_COL].max() if n > 0 else np.nan,
        }

        if n > 1:
            try:
                fit = fit_linear_regression(panel_df[X_COL].values, panel_df[cfg["ycol_new"]].values)
                row["slope"] = fit["slope"]
                row["intercept"] = fit["intercept"]
                row["R2"] = fit["r2"]
                row["p_value"] = fit["p"]
            except Exception:
                pass

        rows.append(row)

    return pd.DataFrame(rows)


def collect_paired_valid_results(df: pd.DataFrame, model_configs: list) -> pd.DataFrame:
    rows = []

    id_candidates = ["SITE_ID", "site_id", "Site_ID", "site_name", "SITE_NAME", "Number", "NUMBER"]
    keep_id_cols = [c for c in id_candidates if c in df.columns]

    for cfg in model_configs:
        baseline_col = cfg["baseline_col"]
        improved_col = cfg["improved_col"]

        baseline = clean_metric_series(df[baseline_col])
        improved = clean_metric_series(df[improved_col])

        valid_pair = (
            baseline.notna()
            & improved.notna()
            & (baseline >= KGE_THRESHOLD)
            & (improved >= KGE_THRESHOLD)
        )

        if valid_pair.sum() == 0:
            continue

        base_r_col = guess_component_col(df, baseline_col, "r")
        imp_r_col = guess_component_col(df, improved_col, "r")
        base_gamma_col = guess_component_col(df, baseline_col, "gamma")
        imp_gamma_col = guess_component_col(df, improved_col, "gamma")
        base_beta_col = guess_component_col(df, baseline_col, "beta")
        imp_beta_col = guess_component_col(df, improved_col, "beta")

        sub = df.loc[valid_pair].copy()
        out = pd.DataFrame(index=sub.index)

        for c in keep_id_cols:
            out[c] = sub[c]

        if IGBP_COL in sub.columns:
            out[IGBP_COL] = sub[IGBP_COL]

        if X_COL in sub.columns:
            out[X_COL] = pd.to_numeric(sub[X_COL], errors="coerce")

        out["Model"] = cfg["name"]
        out["baseline_col"] = baseline_col
        out["improved_col"] = improved_col

        out["KGE_prime_baseline"] = clean_metric_series(sub[baseline_col])
        out["KGE_prime_improved"] = clean_metric_series(sub[improved_col])
        out["delta_KGE_prime"] = out["KGE_prime_improved"] - out["KGE_prime_baseline"]

        out["r_baseline"] = clean_metric_series(sub[base_r_col]) if base_r_col else np.nan
        out["r_improved"] = clean_metric_series(sub[imp_r_col]) if imp_r_col else np.nan
        out["delta_r"] = out["r_improved"] - out["r_baseline"]

        out["gamma_baseline"] = clean_metric_series(sub[base_gamma_col]) if base_gamma_col else np.nan
        out["gamma_improved"] = clean_metric_series(sub[imp_gamma_col]) if imp_gamma_col else np.nan
        out["delta_gamma"] = out["gamma_improved"] - out["gamma_baseline"]

        out["beta_baseline"] = clean_metric_series(sub[base_beta_col]) if base_beta_col else np.nan
        out["beta_improved"] = clean_metric_series(sub[imp_beta_col]) if imp_beta_col else np.nan
        out["delta_beta"] = out["beta_improved"] - out["beta_baseline"]

        rows.append(out)

    if len(rows) == 0:
        return pd.DataFrame()

    paired_df = pd.concat(rows, axis=0, ignore_index=True)

    first_cols = [c for c in ["NUMBER", "Number", "SITE_ID", "site_id", "Site_ID", "IGBP", X_COL, "Model"] if c in paired_df.columns]
    other_cols = [c for c in paired_df.columns if c not in first_cols]
    paired_df = paired_df[first_cols + other_cols]

    return paired_df


def plot_panel(ax, panel_df: pd.DataFrame, cfg: dict, add_legend: bool = False):
    # scatter by biome
    for group in GROUP_ORDER:
        sub = panel_df.loc[panel_df["group_label"] == group].copy()
        if sub.empty:
            continue

        ax.scatter(
            sub[X_COL],
            sub[cfg["ycol_new"]],
            marker="o",
            s=POINT_SIZE,
            linewidths=0.5,
            edgecolors="black",
            alpha=POINT_ALPHA,
            color=GROUP_COLOR_MAP[group],
            label=group,
            zorder=3
        )

    # zero line
    ax.axhline(0, color="gray", linestyle="-.", linewidth=1.0, zorder=1)

    # global regression
    if len(panel_df) > 1:
        try:
            fit_all = fit_linear_regression(panel_df[X_COL].values, panel_df[cfg["ycol_new"]].values)

            ax.fill_between(
                fit_all["x_grid"],
                fit_all["ci_low"],
                fit_all["ci_high"],
                color="lightgray",
                alpha=0.45,
                zorder=1
            )

            ax.plot(
                fit_all["x_grid"],
                fit_all["y_fit"],
                color="black",
                linestyle="--",
                linewidth=2.0,
                zorder=5
            )
        except Exception:
            pass

    # biome regressions
    for group in GROUP_ORDER:
        sub = panel_df.loc[panel_df["group_label"] == group].copy()
        if len(sub) <= 1:
            continue

        try:
            fit = fit_linear_regression(sub[X_COL].values, sub[cfg["ycol_new"]].values)
            line_color = GROUP_COLOR_MAP[group]
            fill_color = to_rgba(line_color, alpha=0.22)

            ax.fill_between(
                fit["x_grid"],
                fit["ci_low"],
                fit["ci_high"],
                color=fill_color,
                zorder=2
            )

            ax.plot(
                fit["x_grid"],
                fit["y_fit"],
                color=line_color,
                linestyle="-",
                linewidth=2.0,
                zorder=4
            )
        except Exception:
            continue

    # panel label outside top-left
    ax.text(
        0.0, 1.04,
        cfg["panel"],
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=14,
        clip_on=False
    )

    # model name inside panel
    ax.text(
        0.03, 0.97,
        cfg["name"],
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14
    )

    ax.set_xlabel("Dryness index", fontsize=14)
    ax.set_ylabel(cfg["ylabel"], fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)

    if add_legend:
        ax.legend(
            loc="lower right",
            frameon=False,
            fontsize=10,
            handlelength=1.6,
            borderaxespad=0.3,
            ncol=1
        )


def get_global_y_limits(df: pd.DataFrame, model_configs: list):
    y_values = []

    for cfg in model_configs:
        panel_df = prepare_panel_data(df, cfg["ycol_new"])
        y_values.extend(panel_df[cfg["ycol_new"]].dropna().tolist())

    if len(y_values) == 0:
        return -0.2, 0.2

    y_min = np.nanmin(y_values)
    y_max = np.nanmax(y_values)
    pad = 0.08 * (y_max - y_min) if y_max > y_min else 0.1

    return y_min - pad, y_max + pad


def print_biome_counts(df: pd.DataFrame, model_configs: list, title: str):
    print(f"\n[INFO] Valid plotted rows by model and biome: {title}")
    out_rows = []

    for cfg in model_configs:
        panel_df = prepare_panel_data(df, cfg["ycol_new"])
        count_s = panel_df.groupby("group_label").size()
        for biome, n in count_s.items():
            out_rows.append({"Model": cfg["name"], "Biome": biome, "N": int(n)})

    if out_rows:
        count_df = pd.DataFrame(out_rows)
        print(count_df.pivot(index="Biome", columns="Model", values="N").fillna(0).astype(int))
    else:
        print("No valid rows for plotting.")


def process_dataset(input_file, model_configs, stat_file, paired_file, biome_reg_file, global_reg_file, tag):
    print(f"\n{'=' * 70}")
    print(f"[INFO] Processing {tag}")
    print(f"{'=' * 70}")

    df = load_data(input_file)
    check_required_columns(df, model_configs)

    df = standardize_igbp(df)
    df = exclude_igbp_classes(df, EXCLUDE_IGBP)
    df = add_delta_columns(df, model_configs)

    stat_df = collect_statistics(df, model_configs)
    stat_df.to_csv(stat_file, index=False)
    print(f"[OK] Statistics saved to: {stat_file}")

    paired_df = collect_paired_valid_results(df, model_configs)
    paired_df.to_csv(paired_file, index=False)
    print(f"[OK] Paired valid-site table saved to: {paired_file}")

    biome_reg_df = collect_biome_regression_statistics(df, model_configs)
    biome_reg_df.to_csv(biome_reg_file, index=False)
    print(f"[OK] Biome regression statistics saved to: {biome_reg_file}")

    global_reg_df = collect_global_regression_statistics(df, model_configs)
    global_reg_df.to_csv(global_reg_file, index=False)
    print(f"[OK] Global regression statistics saved to: {global_reg_file}")

    print_biome_counts(df, model_configs, title=tag)

    return df


def main():
    # =========================
    # Process gs
    # =========================
    df_gs = process_dataset(
        input_file=gs_INPUT_FILE,
        model_configs=MODEL_CONFIGS_gs,
        stat_file=gs_STAT_FILE,
        paired_file=gs_PAIRED_FILE,
        biome_reg_file=gs_BIOME_REG_STAT_FILE,
        global_reg_file=gs_GLOBAL_REG_STAT_FILE,
        tag="gs"
    )

    # =========================
    # Process ET
    # =========================
    df_ET = process_dataset(
        input_file=ET_INPUT_FILE,
        model_configs=MODEL_CONFIGS_ET,
        stat_file=ET_STAT_FILE,
        paired_file=ET_PAIRED_FILE,
        biome_reg_file=ET_BIOME_REG_STAT_FILE,
        global_reg_file=ET_GLOBAL_REG_STAT_FILE,
        tag="ET"
    )

    # =========================
    # Figure: 4 rows x 2 cols
    # =========================
    fig, axes = plt.subplots(4, 2, figsize=(14, 18))

    y_low_gs, y_high_gs = get_global_y_limits(df_gs, MODEL_CONFIGS_gs)
    y_low_ET, y_high_ET = get_global_y_limits(df_ET, MODEL_CONFIGS_ET)

    for i in range(4):
        # Left column: gs
        cfg_gs = MODEL_CONFIGS_gs[i]
        panel_df_gs = prepare_panel_data(df_gs, cfg_gs["ycol_new"])
        plot_panel(
            ax=axes[i, 0],
            panel_df=panel_df_gs,
            cfg=cfg_gs,
            add_legend=(i == 0)
        )
        axes[i, 0].set_ylim(y_low_gs, y_high_gs)

        # Right column: ET
        cfg_ET = MODEL_CONFIGS_ET[i]
        panel_df_ET = prepare_panel_data(df_ET, cfg_ET["ycol_new"])
        plot_panel(
            ax=axes[i, 1],
            panel_df=panel_df_ET,
            cfg=cfg_ET,
            add_legend=False
        )
        axes[i, 1].set_ylim(y_low_ET, y_high_ET)

    # Column titles
    axes[0, 0].set_title(r"$g_c$", fontsize=16, pad=30)
    axes[0, 1].set_title(r"$ET$", fontsize=16, pad=30)

    plt.tight_layout(pad=1.8, h_pad=1.4, w_pad=1.2)
    fig.savefig(FIG_FILE, dpi=300, bbox_inches="tight")
    print(f"[OK] Figure saved to: {FIG_FILE}")

    plt.show()


if __name__ == "__main__":
    main()