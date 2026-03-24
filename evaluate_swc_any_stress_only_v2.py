
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from utils_metrics import calculate_kge_prime


# =========================================================
# Paths
# =========================================================
BASE_INPUT_DIR = Path("./fluxnet_model_input")
BASE_OUTPUT_DIR = Path("./fluxnet_model_output")
RESULT_DIR = BASE_OUTPUT_DIR / "stress_evaluation_any_stress"


# =========================================================
# User settings
# =========================================================
SCALES = ["DD", "HH"]

# Site-specific stress thresholds
VPD_PERCENTILE = 90
SWC_PERCENTILE = 10

# Which SWC variable to use for stress classification:
#   "layer_1"      -> always use SWC_layer_1
#   "profile_mean" -> use SWC_profile_mean only
#   "auto"         -> use SWC_profile_mean when available, otherwise SWC_layer_1
STRESS_SWC_MODE = "layer_1"

# Minimum valid sample size BEFORE extracting any-stress records
# IMPORTANT:
#   - Only DD is used to determine whether a site is eligible
#   - If a site passes DD >= 500 for a given variable/comparison,
#     the corresponding HH site is retained as well
MIN_DD_TOTAL_N = 500


# =========================================================
# Silence warnings
# =========================================================
warnings.filterwarnings("ignore", message="Could not infer format", category=UserWarning)
warnings.filterwarnings("ignore", message="invalid value encountered", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="divide by zero encountered", category=RuntimeWarning)


# =========================================================
# Helpers
# =========================================================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_site_data(file_path: Path, scale: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, low_memory=False)
    df.columns = df.columns.str.strip()

    if "TIMESTAMP" not in df.columns:
        raise KeyError(f"{file_path.name}: missing required column 'TIMESTAMP'")

    df["TIMESTAMP"] = df["TIMESTAMP"].astype(str).str.strip()
    df = df[df["TIMESTAMP"].str.lower() != "dimensionless"].copy()

    if scale == "DD":
        df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], format="%Y%m%d", errors="coerce")
    elif scale == "HH":
        df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], format="%Y%m%d%H%M", errors="coerce")
    else:
        raise ValueError(f"Unsupported scale: {scale}")

    df = df[df["TIMESTAMP"].notna()].copy()

    for col in df.columns:
        if col != "TIMESTAMP":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "gs" in df.columns and "gs_obs" not in df.columns:
        df = df.rename(columns={"gs": "gs_obs"})

    return df.reset_index(drop=True)


def safe_metrics(obs, pred):
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(pred)

    if mask.sum() < 2:
        return (np.nan, np.nan, np.nan, np.nan)

    try:
        with np.errstate(invalid="ignore", divide="ignore"):
            return calculate_kge_prime(obs[mask], pred[mask])
    except Exception:
        return (np.nan, np.nan, np.nan, np.nan)


# =========================================================
# Stress classification
# =========================================================
def choose_stress_swc_column(df: pd.DataFrame) -> str:
    if STRESS_SWC_MODE == "layer_1":
        return "SWC_layer_1"
    if STRESS_SWC_MODE == "profile_mean":
        return "SWC_profile_mean"
    if STRESS_SWC_MODE == "auto":
        if "SWC_profile_mean" in df.columns and df["SWC_profile_mean"].notna().sum() >= 10:
            return "SWC_profile_mean"
        return "SWC_layer_1"
    raise ValueError(f"Unsupported STRESS_SWC_MODE: {STRESS_SWC_MODE}")


def add_any_stress_flag(full_df: pd.DataFrame, merged_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    swc_col = choose_stress_swc_column(full_df)

    if "VPD" not in full_df.columns:
        raise KeyError("Missing required column 'VPD' in input data.")
    if swc_col not in full_df.columns:
        raise KeyError(f"Missing required column '{swc_col}' in input data.")

    vpd_series = pd.to_numeric(full_df["VPD"], errors="coerce")
    swc_series = pd.to_numeric(full_df[swc_col], errors="coerce")

    vpd_p90 = np.nanpercentile(vpd_series.to_numpy(dtype=float), VPD_PERCENTILE)
    swc_p10 = np.nanpercentile(swc_series.to_numpy(dtype=float), SWC_PERCENTILE)

    merged_df = merged_df.copy()
    merged_df["stress_swc_col"] = swc_col
    merged_df["stress_vpd_p90"] = vpd_p90
    merged_df["stress_swc_p10"] = swc_p10

    vpd = pd.to_numeric(merged_df["VPD"], errors="coerce")
    swc = pd.to_numeric(merged_df[swc_col], errors="coerce")

    high_vpd = vpd > vpd_p90
    low_swc = swc < swc_p10

    merged_df["any_stress"] = high_vpd | low_swc

    meta = {
        "stress_swc_col": swc_col,
        "stress_vpd_p90": vpd_p90,
        "stress_swc_p10": swc_p10,
    }
    return merged_df, meta


# =========================================================
# Comparison definitions
# =========================================================
def get_gs_comparisons():
    return [
        {"comparison": "BBL_mSWC_layer_1_vs_noSWC", "base_col": "BBL_pred_noSWC", "constrained_col": "BBL_pred_mSWC_layer_1"},
        {"comparison": "BBL_mSWC_profile_mean_vs_noSWC", "base_col": "BBL_pred_noSWC", "constrained_col": "BBL_pred_mSWC_profile_mean"},
        {"comparison": "Medlyn_mSWC_layer_1_vs_noSWC", "base_col": "Medlyn_pred_noSWC", "constrained_col": "Medlyn_pred_mSWC_layer_1"},
        {"comparison": "Medlyn_mSWC_profile_mean_vs_noSWC", "base_col": "Medlyn_pred_noSWC", "constrained_col": "Medlyn_pred_mSWC_profile_mean"},
        {"comparison": "RF_GPP_VPD_leaf_mSWC_layer_1_vs_noSWC", "base_col": "RF_GPP_VPD_leaf_pred_noSWC", "constrained_col": "RF_GPP_VPD_leaf_pred_mSWC_layer_1"},
        {"comparison": "RF_GPP_VPD_leaf_mSWC_profile_mean_vs_noSWC", "base_col": "RF_GPP_VPD_leaf_pred_noSWC", "constrained_col": "RF_GPP_VPD_leaf_pred_mSWC_profile_mean"},
        {"comparison": "RF_direct_SWC_layer_1_vs_noSWC", "base_col": "RF_GPP_VPD_leaf_pred_noSWC", "constrained_col": "RF_GPP_VPD_leaf_SWC_layer_1_pred"},
        {"comparison": "RF_direct_SWC_profile_mean_vs_noSWC", "base_col": "RF_GPP_VPD_leaf_pred_noSWC", "constrained_col": "RF_GPP_VPD_leaf_SWC_profile_mean_pred"},
    ]


def get_et_comparisons():
    return [
        {"comparison": "BBL_mSWC_layer_1_vs_noSWC", "base_col": "BBL_ET_pred_noSWC", "constrained_col": "BBL_ET_pred_mSWC_layer_1"},
        {"comparison": "BBL_mSWC_profile_mean_vs_noSWC", "base_col": "BBL_ET_pred_noSWC", "constrained_col": "BBL_ET_pred_mSWC_profile_mean"},
        {"comparison": "Medlyn_mSWC_layer_1_vs_noSWC", "base_col": "Medlyn_ET_pred_noSWC", "constrained_col": "Medlyn_ET_pred_mSWC_layer_1"},
        {"comparison": "Medlyn_mSWC_profile_mean_vs_noSWC", "base_col": "Medlyn_ET_pred_noSWC", "constrained_col": "Medlyn_ET_pred_mSWC_profile_mean"},
        {"comparison": "RF_GPP_VPD_leaf_mSWC_layer_1_vs_noSWC", "base_col": "RF_GPP_VPD_leaf_ET_pred_noSWC", "constrained_col": "RF_GPP_VPD_leaf_ET_pred_mSWC_layer_1"},
        {"comparison": "RF_GPP_VPD_leaf_mSWC_profile_mean_vs_noSWC", "base_col": "RF_GPP_VPD_leaf_ET_pred_noSWC", "constrained_col": "RF_GPP_VPD_leaf_ET_pred_mSWC_profile_mean"},
        {"comparison": "RF_direct_SWC_layer_1_vs_noSWC", "base_col": "RF_GPP_VPD_leaf_ET_pred_noSWC", "constrained_col": "RF_GPP_VPD_leaf_SWC_layer_1_ET_pred"},
        {"comparison": "RF_direct_SWC_profile_mean_vs_noSWC", "base_col": "RF_GPP_VPD_leaf_ET_pred_noSWC", "constrained_col": "RF_GPP_VPD_leaf_SWC_profile_mean_ET_pred"},
    ]


# =========================================================
# Evaluation
# =========================================================
def count_valid_triplets(df: pd.DataFrame, obs_col: str, base_col: str, constrained_col: str) -> int:
    required = [obs_col, base_col, constrained_col]
    if any(col not in df.columns for col in required):
        return 0

    valid_total = (
        pd.to_numeric(df[obs_col], errors="coerce").notna()
        & pd.to_numeric(df[base_col], errors="coerce").notna()
        & pd.to_numeric(df[constrained_col], errors="coerce").notna()
    )
    return int(valid_total.sum())


def evaluate_comparison(df: pd.DataFrame, obs_col: str, base_col: str, constrained_col: str) -> dict | None:
    required = [obs_col, base_col, constrained_col, "any_stress"]
    if any(col not in df.columns for col in required):
        return None

    valid_total = (
        pd.to_numeric(df[obs_col], errors="coerce").notna()
        & pd.to_numeric(df[base_col], errors="coerce").notna()
        & pd.to_numeric(df[constrained_col], errors="coerce").notna()
    )
    n_total = int(valid_total.sum())

    cond_mask = valid_total & df["any_stress"].fillna(False).astype(bool)
    n_any_stress = int(cond_mask.sum())

    if n_any_stress < 2:
        return None

    obs = pd.to_numeric(df.loc[cond_mask, obs_col], errors="coerce").to_numpy(dtype=float)
    base_pred = pd.to_numeric(df.loc[cond_mask, base_col], errors="coerce").to_numpy(dtype=float)
    constrained_pred = pd.to_numeric(df.loc[cond_mask, constrained_col], errors="coerce").to_numpy(dtype=float)

    base_metrics = safe_metrics(obs, base_pred)
    constrained_metrics = safe_metrics(obs, constrained_pred)

    return {
        "n_total_before_filter": n_total,
        "n_any_stress": n_any_stress,
        "base_KGE_prime": base_metrics[0],
        "base_r": base_metrics[1],
        "base_gamma": base_metrics[2],
        "base_beta": base_metrics[3],
        "constrained_KGE_prime": constrained_metrics[0],
        "constrained_r": constrained_metrics[1],
        "constrained_gamma": constrained_metrics[2],
        "constrained_beta": constrained_metrics[3],
        "delta_KGE_prime": constrained_metrics[0] - base_metrics[0],
        "delta_r": constrained_metrics[1] - base_metrics[1],
        "delta_gamma": constrained_metrics[2] - base_metrics[2],
        "delta_beta": constrained_metrics[3] - base_metrics[3],
        "improved_KGE_prime": int(
            np.isfinite(constrained_metrics[0] - base_metrics[0])
            and (constrained_metrics[0] > base_metrics[0])
        ),
    }


def summarize_results(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        return pd.DataFrame()

    def safe_median(x):
        x = pd.to_numeric(x, errors="coerce")
        x = x[np.isfinite(x)]
        return np.nan if len(x) == 0 else float(np.median(x))

    def safe_mean(x):
        x = pd.to_numeric(x, errors="coerce")
        x = x[np.isfinite(x)]
        return np.nan if len(x) == 0 else float(np.mean(x))

    grouped = []
    for keys, sub in detail_df.groupby(["scale", "variable", "comparison"], dropna=False):
        grouped.append(
            {
                "scale": keys[0],
                "variable": keys[1],
                "comparison": keys[2],
                "n_sites": len(sub),
                "sum_any_stress_records": int(pd.to_numeric(sub["n_any_stress"], errors="coerce").fillna(0).sum()),
                "median_n_any_stress": safe_median(sub["n_any_stress"]),
                "mean_n_any_stress": safe_mean(sub["n_any_stress"]),
                "n_sites_improved": int(pd.to_numeric(sub["improved_KGE_prime"], errors="coerce").fillna(0).sum()),
                "fraction_sites_improved": safe_mean(sub["improved_KGE_prime"]),
                "median_base_KGE_prime": safe_median(sub["base_KGE_prime"]),
                "median_constrained_KGE_prime": safe_median(sub["constrained_KGE_prime"]),
                "median_delta_KGE_prime": safe_median(sub["delta_KGE_prime"]),
                "mean_delta_KGE_prime": safe_mean(sub["delta_KGE_prime"]),
                "median_delta_r": safe_median(sub["delta_r"]),
                "median_delta_gamma": safe_median(sub["delta_gamma"]),
                "median_delta_beta": safe_median(sub["delta_beta"]),
            }
        )

    return pd.DataFrame(grouped).sort_values(["scale", "variable", "comparison"], ignore_index=True)


def summarize_eligible_sites(site_record_df: pd.DataFrame) -> pd.DataFrame:
    if site_record_df.empty:
        return pd.DataFrame()

    out = (
        site_record_df.groupby(["scale", "variable"], dropna=False)
        .agg(
            n_site_combinations=("site_stem", "size"),
            n_unique_sites=("site_stem", "nunique"),
            median_n_total_before_filter=("n_total_before_filter", "median"),
            median_n_any_stress=("n_any_stress", "median"),
            mean_n_any_stress=("n_any_stress", "mean"),
        )
        .reset_index()
    )
    return out.sort_values(["scale", "variable"], ignore_index=True)


# =========================================================
# Build merged data
# =========================================================
def build_merged_frames(scale: str, site_stem: str, gs_pred_path: Path, et_pred_path: Path):
    input_path = BASE_INPUT_DIR / scale / f"{site_stem}.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    full_df = load_site_data(input_path, scale)
    gs_df = pd.read_csv(gs_pred_path)
    et_df = pd.read_csv(et_pred_path)

    gs_df["TIMESTAMP"] = pd.to_datetime(gs_df["TIMESTAMP"], errors="coerce")
    et_df["TIMESTAMP"] = pd.to_datetime(et_df["TIMESTAMP"], errors="coerce")

    merge_cols = ["TIMESTAMP", "VPD", "SWC_layer_1"]
    if "SWC_profile_mean" in full_df.columns:
        merge_cols.append("SWC_profile_mean")

    full_small = full_df[merge_cols].drop_duplicates(subset=["TIMESTAMP"])

    gs_merged = gs_df.merge(full_small, on="TIMESTAMP", how="left", suffixes=("", "_input"))
    et_merged = et_df.merge(full_small, on="TIMESTAMP", how="left", suffixes=("", "_input"))

    gs_merged, stress_meta = add_any_stress_flag(full_df, gs_merged)
    et_merged, _ = add_any_stress_flag(full_df, et_merged)

    return gs_merged, et_merged, stress_meta


# =========================================================
# DD eligibility
# =========================================================
def collect_dd_eligibility():
    dd_gs_pred_dir = BASE_OUTPUT_DIR / "DD" / "gs_predictions"
    dd_et_pred_dir = BASE_OUTPUT_DIR / "DD" / "ET_predictions"

    if not dd_gs_pred_dir.exists() or not dd_et_pred_dir.exists():
        raise FileNotFoundError("DD prediction directories are required to determine site eligibility.")

    eligible_keys = set()
    dd_site_records = []

    gs_pred_files = sorted(dd_gs_pred_dir.glob("*_gs_predictions.csv"))

    for gs_pred_path in gs_pred_files:
        site_stem = gs_pred_path.name.replace("_gs_predictions.csv", "")
        et_pred_path = dd_et_pred_dir / f"{site_stem}_ET_predictions.csv"
        if not et_pred_path.exists():
            continue

        try:
            gs_merged, et_merged, stress_meta = build_merged_frames("DD", site_stem, gs_pred_path, et_pred_path)
        except Exception as e:
            print(f"[WARN] DD eligibility skipped | {site_stem} | {e}")
            continue

        for cmp_cfg in get_gs_comparisons():
            n_total = count_valid_triplets(gs_merged, "gs_obs", cmp_cfg["base_col"], cmp_cfg["constrained_col"])
            passed = int(n_total >= MIN_DD_TOTAL_N)
            if passed:
                eligible_keys.add(("gs", cmp_cfg["comparison"], site_stem))
            dd_site_records.append(
                {
                    "site_stem": site_stem,
                    "scale": "DD",
                    "variable": "gs",
                    "comparison": cmp_cfg["comparison"],
                    "n_total_before_filter": n_total,
                    "passed_dd_threshold": passed,
                    "stress_swc_col": stress_meta["stress_swc_col"],
                    "stress_vpd_p90": stress_meta["stress_vpd_p90"],
                    "stress_swc_p10": stress_meta["stress_swc_p10"],
                }
            )

        for cmp_cfg in get_et_comparisons():
            n_total = count_valid_triplets(et_merged, "ET_obs", cmp_cfg["base_col"], cmp_cfg["constrained_col"])
            passed = int(n_total >= MIN_DD_TOTAL_N)
            if passed:
                eligible_keys.add(("ET", cmp_cfg["comparison"], site_stem))
            dd_site_records.append(
                {
                    "site_stem": site_stem,
                    "scale": "DD",
                    "variable": "ET",
                    "comparison": cmp_cfg["comparison"],
                    "n_total_before_filter": n_total,
                    "passed_dd_threshold": passed,
                    "stress_swc_col": stress_meta["stress_swc_col"],
                    "stress_vpd_p90": stress_meta["stress_vpd_p90"],
                    "stress_swc_p10": stress_meta["stress_swc_p10"],
                }
            )

    return eligible_keys, pd.DataFrame(dd_site_records)


# =========================================================
# Per-site processing
# =========================================================
def process_site(scale: str, gs_pred_path: Path, et_pred_path: Path, eligible_keys: set):
    site_stem = gs_pred_path.name.replace("_gs_predictions.csv", "")
    gs_merged, et_merged, stress_meta = build_merged_frames(scale, site_stem, gs_pred_path, et_pred_path)

    gs_rows = []
    et_rows = []
    eligible_site_rows = []

    for cmp_cfg in get_gs_comparisons():
        key = ("gs", cmp_cfg["comparison"], site_stem)
        if key not in eligible_keys:
            continue

        out = evaluate_comparison(
            df=gs_merged,
            obs_col="gs_obs",
            base_col=cmp_cfg["base_col"],
            constrained_col=cmp_cfg["constrained_col"],
        )
        if out is None:
            continue

        row = {
            "site_stem": site_stem,
            "scale": scale,
            "variable": "gs",
            "comparison": cmp_cfg["comparison"],
            **stress_meta,
            **out,
        }
        gs_rows.append(row)
        eligible_site_rows.append(
            {
                "site_stem": site_stem,
                "scale": scale,
                "variable": "gs",
                "comparison": cmp_cfg["comparison"],
                "eligibility_source": "DD>=500",
                "n_total_before_filter": out["n_total_before_filter"],
                "n_any_stress": out["n_any_stress"],
                "stress_swc_col": stress_meta["stress_swc_col"],
                "stress_vpd_p90": stress_meta["stress_vpd_p90"],
                "stress_swc_p10": stress_meta["stress_swc_p10"],
            }
        )

    for cmp_cfg in get_et_comparisons():
        key = ("ET", cmp_cfg["comparison"], site_stem)
        if key not in eligible_keys:
            continue

        out = evaluate_comparison(
            df=et_merged,
            obs_col="ET_obs",
            base_col=cmp_cfg["base_col"],
            constrained_col=cmp_cfg["constrained_col"],
        )
        if out is None:
            continue

        row = {
            "site_stem": site_stem,
            "scale": scale,
            "variable": "ET",
            "comparison": cmp_cfg["comparison"],
            **stress_meta,
            **out,
        }
        et_rows.append(row)
        eligible_site_rows.append(
            {
                "site_stem": site_stem,
                "scale": scale,
                "variable": "ET",
                "comparison": cmp_cfg["comparison"],
                "eligibility_source": "DD>=500",
                "n_total_before_filter": out["n_total_before_filter"],
                "n_any_stress": out["n_any_stress"],
                "stress_swc_col": stress_meta["stress_swc_col"],
                "stress_vpd_p90": stress_meta["stress_vpd_p90"],
                "stress_swc_p10": stress_meta["stress_swc_p10"],
            }
        )

    return gs_rows, et_rows, eligible_site_rows


def run_scale(scale: str, eligible_keys: set, dd_site_record_df: pd.DataFrame):
    scale_result_dir = RESULT_DIR / scale
    ensure_dir(scale_result_dir)

    gs_pred_dir = BASE_OUTPUT_DIR / scale / "gs_predictions"
    et_pred_dir = BASE_OUTPUT_DIR / scale / "ET_predictions"

    if not gs_pred_dir.exists() or not et_pred_dir.exists():
        print(f"[WARN] Missing prediction directory for {scale}.")
        return

    gs_pred_files = sorted(gs_pred_dir.glob("*_gs_predictions.csv"))
    print(f"[INFO] {scale}: found {len(gs_pred_files)} gs prediction files")

    all_gs_rows = []
    all_et_rows = []
    eligible_site_rows = []

    for gs_pred_path in gs_pred_files:
        site_stem = gs_pred_path.name.replace("_gs_predictions.csv", "")
        et_pred_path = et_pred_dir / f"{site_stem}_ET_predictions.csv"

        if not et_pred_path.exists():
            print(f"[WARN] Missing ET prediction file for {scale} | {site_stem}")
            continue

        try:
            gs_rows, et_rows, site_rows = process_site(scale, gs_pred_path, et_pred_path, eligible_keys)
            all_gs_rows.extend(gs_rows)
            all_et_rows.extend(et_rows)
            eligible_site_rows.extend(site_rows)
            print(f"[INFO] Processed {scale} | {site_stem}")
        except Exception as e:
            print(f"[ERROR] {scale} | {site_stem} | {e}")

    gs_detail = pd.DataFrame(all_gs_rows)
    et_detail = pd.DataFrame(all_et_rows)
    eligible_sites = pd.DataFrame(eligible_site_rows)

    gs_summary = summarize_results(gs_detail)
    et_summary = summarize_results(et_detail)
    eligible_summary = summarize_eligible_sites(eligible_sites)

    gs_detail.to_csv(scale_result_dir / f"{scale}_any_stress_gs_site_combination.csv", index=False)
    et_detail.to_csv(scale_result_dir / f"{scale}_any_stress_ET_site_combination.csv", index=False)
    gs_summary.to_csv(scale_result_dir / f"{scale}_any_stress_gs_summary.csv", index=False)
    et_summary.to_csv(scale_result_dir / f"{scale}_any_stress_ET_summary.csv", index=False)

    eligible_sites.to_csv(scale_result_dir / f"{scale}_eligible_site_record.csv", index=False)
    eligible_summary.to_csv(scale_result_dir / f"{scale}_eligible_site_summary.csv", index=False)

    if scale == "DD":
        dd_site_record_df.to_csv(scale_result_dir / f"{scale}_dd_threshold_check_record.csv", index=False)
        dd_passed = dd_site_record_df[dd_site_record_df["passed_dd_threshold"] == 1].copy()
        dd_passed.to_csv(scale_result_dir / f"{scale}_eligible_from_dd_threshold.csv", index=False)


def main():
    ensure_dir(RESULT_DIR)
    eligible_keys, dd_site_record_df = collect_dd_eligibility()

    for scale in SCALES:
        run_scale(scale, eligible_keys, dd_site_record_df)

    print(f"[DONE] Outputs written to: {RESULT_DIR.resolve()}")


if __name__ == "__main__":
    main()
