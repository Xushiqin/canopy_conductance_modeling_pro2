from pathlib import Path
import traceback
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from utils_io import split_train_test, ensure_dir
from utils_metrics import calculate_kge_prime
from utils_smc import fit_n_parameter, calculate_m_sm

from model_bbl import fit_and_predict as run_bbl
from model_medlyn import fit_and_predict as run_medlyn
from model_rf_gpp_vpdleaf import fit_and_predict as run_rf_gpp_vpd
from model_pm import estimate_et_pm, get_observed_et


BASE_INPUT_DIR = Path("./fluxnet_model_input")
BASE_OUTPUT_DIR = Path("./fluxnet_model_output")
BIF_MODEL_SITES_FILE = Path("./fluxnet/BIF_summary/BIF_model_sites.csv")
SCALES = ["DD", "HH"]

RF_RANDOM_STATE = 42


warnings.filterwarnings("ignore", message="Could not infer format", category=UserWarning)
warnings.filterwarnings("ignore", message="invalid value encountered", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="divide by zero encountered", category=RuntimeWarning)


def extract_site_id_from_filename(file_name: str) -> str:
    parts = str(file_name).split("_")
    return parts[1].strip() if len(parts) >= 3 else ""


def load_bif_model_sites(bif_file: Path) -> pd.DataFrame:
    if not bif_file.exists():
        print(f"[WARN] BIF model sites file not found: {bif_file}")
        return pd.DataFrame(columns=["FILE_NAME", "SITE_ID", "IGBP"])

    df = pd.read_csv(bif_file, dtype=str)
    df.columns = df.columns.str.strip()

    for col in ["FILE_NAME", "SITE_ID", "IGBP"]:
        if col not in df.columns:
            df[col] = ""

    df["FILE_NAME"] = df["FILE_NAME"].astype(str).str.strip()
    df["SITE_ID"] = df["SITE_ID"].astype(str).str.strip()
    df["IGBP"] = df["IGBP"].astype(str).str.strip()

    return df[["FILE_NAME", "SITE_ID", "IGBP"]].drop_duplicates(subset=["SITE_ID"])


def attach_bif_info(row_dict: dict, file_name: str, bif_lookup: pd.DataFrame) -> dict:
    site_id = extract_site_id_from_filename(file_name)
    row_dict["SITE_ID"] = site_id

    matched = bif_lookup.loc[bif_lookup["SITE_ID"] == site_id]
    row_dict["IGBP"] = "" if matched.empty else matched.iloc[0]["IGBP"]
    return row_dict


def load_site_data(file_path: Path, scale: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, low_memory=False)
    df.columns = df.columns.str.strip()

    if "TIMESTAMP" not in df.columns:
        raise KeyError(f"{file_path.name}: missing required column 'TIMESTAMP'")
    if "gs" not in df.columns:
        raise KeyError(f"{file_path.name}: missing required column 'gs'")

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

    df = df.rename(columns={"gs": "gs_obs"}).reset_index(drop=True)

    if df.empty:
        raise ValueError(f"{file_path.name}: no valid data rows after cleaning.")

    return df


def flatten_metrics(prefix: str, metrics: tuple) -> dict:
    return {
        f"{prefix}_KGE_prime": metrics[0],
        f"{prefix}_r": metrics[1],
        f"{prefix}_gamma": metrics[2],
        f"{prefix}_beta": metrics[3],
    }


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


def safe_fit_model(func, train_df, test_df):
    try:
        return func(train_df, test_df, target_col="gs_obs"), None
    except Exception as e:
        return None, str(e)


def extract_first_valid_numeric(series: pd.Series, name: str, file_name: str) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        raise ValueError(f"{file_name}: column '{name}' contains no valid numeric values.")
    return float(values.iloc[0])


def get_numeric_array(df: pd.DataFrame, col: str, fill_nan: bool = False) -> np.ndarray:
    if col not in df.columns:
        if fill_nan:
            return np.full(len(df), np.nan, dtype=float)
        raise KeyError(f"Missing required column: {col}")
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def keep_selected_params(model_name: str, params: dict) -> dict:
    if model_name not in ["BBL", "Medlyn"]:
        return {}
    return {f"{model_name}_{k}": v for k, v in params.items()}


def make_valid_mask(df: pd.DataFrame, feature_cols: list, target_col: str = "gs_obs") -> pd.Series:
    required = feature_cols + [target_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for RF: {missing}")

    tmp = df[required].apply(pd.to_numeric, errors="coerce")
    return tmp.notna().all(axis=1)


def fit_rf_with_features(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list, model_name: str):
    train_mask = make_valid_mask(train_df, feature_cols)
    test_mask = make_valid_mask(test_df, feature_cols)

    train_sub = train_df.loc[train_mask]
    test_sub = test_df.loc[test_mask]

    if len(train_sub) < 10:
        raise ValueError(f"{model_name}: insufficient valid training rows ({len(train_sub)}).")
    if len(test_sub) < 2:
        raise ValueError(f"{model_name}: insufficient valid testing rows ({len(test_sub)}).")

    X_train = train_sub[feature_cols].apply(pd.to_numeric, errors="coerce")
    y_train = pd.to_numeric(train_sub["gs_obs"], errors="coerce")
    X_test = test_sub[feature_cols].apply(pd.to_numeric, errors="coerce")

    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=RF_RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    pred_train = np.full(len(train_df), np.nan, dtype=float)
    pred_test = np.full(len(test_df), np.nan, dtype=float)

    pred_train[train_mask.to_numpy()] = model.predict(X_train)
    pred_test[test_mask.to_numpy()] = model.predict(X_test)

    return {
        "model_name": model_name,
        "params": {"feature_cols": "|".join(feature_cols)},
        "pred_train": pred_train,
        "pred_test": pred_test,
    }


def safe_fit_rf_features(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list, model_name: str):
    try:
        return fit_rf_with_features(train_df, test_df, feature_cols, model_name), None
    except Exception as e:
        return None, str(e)


def evaluate_and_store_standard_model(
    model_name,
    pred_test,
    obs_test,
    et_obs_test,
    test_df,
    scale,
    gs_row,
    et_row,
    gs_export,
    et_export,
):
    gs_metrics = safe_metrics(obs_test, pred_test)
    gs_row.update(flatten_metrics(model_name, gs_metrics))
    gs_export[f"{model_name}_pred"] = pred_test

    et_pred = estimate_et_pm(test_df, pred_test, scale, gs_target_col="gs_obs")
    et_metrics = safe_metrics(et_obs_test, et_pred)
    et_row.update(flatten_metrics(model_name, et_metrics))
    et_export[f"{model_name}_ET_pred"] = et_pred


def run_scale(scale: str, bif_lookup: pd.DataFrame):
    input_dir = BASE_INPUT_DIR / scale
    output_dir = BASE_OUTPUT_DIR / scale

    gs_pred_dir = output_dir / "gs_predictions"
    gs_summary_dir = output_dir / "gs_summary"
    et_pred_dir = output_dir / "ET_predictions"
    et_summary_dir = output_dir / "ET_summary"

    for d in [output_dir, gs_pred_dir, gs_summary_dir, et_pred_dir, et_summary_dir]:
        ensure_dir(d)

    combined_gs_rows = []
    combined_et_rows = []

    if not input_dir.exists():
        print(f"[WARN] Input directory does not exist: {input_dir}")
        return

    site_files = sorted(input_dir.glob("*.csv"))
    print(f"[INFO] {scale}: found {len(site_files)} csv files in {input_dir}")

    for file_path in site_files:
        file_name = file_path.name
        file_stem = file_path.stem
        print(f"[INFO] Processing {scale} | {file_name}")

        try:
            df = load_site_data(file_path, scale)

            required_cols = [
                "TIMESTAMP",
                "gs_obs",
                "SWC_layer_count",
                "SWC_layer_1",
                "s",
                "Rn",
                "ga",
                "TA",
                "VPD",
                "GPP",
                "VPD_leaf",
            ]
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                raise KeyError(f"{file_name}: missing required columns: {missing_cols}")

            train_df, test_df = split_train_test(df)

            if train_df.empty:
                raise ValueError(f"{file_name}: training dataframe is empty.")
            if test_df.empty:
                raise ValueError(f"{file_name}: testing dataframe is empty.")

            swc_layer_count_value = extract_first_valid_numeric(
                train_df["SWC_layer_count"], "SWC_layer_count", file_name
            )

            base_row = attach_bif_info(
                {
                    "FILE_NAME": file_name,
                    "scale": scale,
                    "status": "success",
                    "n_train_records": len(train_df),
                    "n_test_records": len(test_df),
                    "SWC_layer_count": swc_layer_count_value,
                },
                file_name,
                bif_lookup,
            )

            gs_row = base_row.copy()
            et_row = base_row.copy()

            # --------------------------------------------------
            # Framework A: noSWC and mSWC
            # --------------------------------------------------
            no_swc_results = {}
            model_params = {}

            for func in [run_bbl, run_medlyn, run_rf_gpp_vpd]:
                out, err = safe_fit_model(func, train_df, test_df)
                fallback_name = func.__name__.replace("run_", "")

                if out is None:
                    gs_row[f"{fallback_name}_status"] = "failed"
                    gs_row[f"{fallback_name}_error_message"] = err
                    et_row[f"{fallback_name}_status"] = "failed"
                    et_row[f"{fallback_name}_error_message"] = err
                    continue

                model_name = out["model_name"]
                no_swc_results[model_name] = out
                model_params[model_name] = out["params"]
                gs_row[f"{model_name}_status"] = "success"
                et_row[f"{model_name}_status"] = "success"

            if not no_swc_results:
                raise ValueError("All noSWC models failed.")

            pred_train_dict = {k: v["pred_train"] for k, v in no_swc_results.items()}

            n_layer_1 = fit_n_parameter(
                train_df,
                pred_train_dict,
                swc_col="SWC_layer_1",
                target_col="gs_obs",
                step=1.0,
            )
            gs_row["n_layer_1"] = n_layer_1
            et_row["n_layer_1"] = n_layer_1

            m_layer_1_test = (
                calculate_m_sm(
                    get_numeric_array(test_df, "SWC_layer_1"),
                    n_layer_1,
                    get_numeric_array(train_df, "SWC_layer_1"),
                )
                if np.isfinite(n_layer_1)
                else np.full(len(test_df), np.nan, dtype=float)
            )

            profile_available = (
                "SWC_profile_mean" in train_df.columns
                and pd.to_numeric(train_df["SWC_profile_mean"], errors="coerce").notna().sum() >= 2
                and pd.to_numeric(test_df["SWC_profile_mean"], errors="coerce").notna().sum() >= 2
                and swc_layer_count_value >= 2
            )

            if profile_available:
                n_profile_mean = fit_n_parameter(
                    train_df,
                    pred_train_dict,
                    swc_col="SWC_profile_mean",
                    target_col="gs_obs",
                    step=1.0,
                )
                gs_row["n_profile_mean"] = n_profile_mean
                et_row["n_profile_mean"] = n_profile_mean

                m_profile_test = (
                    calculate_m_sm(
                        get_numeric_array(test_df, "SWC_profile_mean"),
                        n_profile_mean,
                        get_numeric_array(train_df, "SWC_profile_mean"),
                    )
                    if np.isfinite(n_profile_mean)
                    else np.full(len(test_df), np.nan, dtype=float)
                )
            else:
                gs_row["n_profile_mean"] = np.nan
                et_row["n_profile_mean"] = np.nan
                m_profile_test = np.full(len(test_df), np.nan, dtype=float)

            obs_test = get_numeric_array(test_df, "gs_obs")
            et_obs_test = get_observed_et(test_df, scale)

            for model_name, params in model_params.items():
                gs_row.update(keep_selected_params(model_name, params))
                et_row.update(keep_selected_params(model_name, params))

            gs_export = pd.DataFrame(
                {
                    "TIMESTAMP": test_df["TIMESTAMP"].values,
                    "gs_obs": obs_test,
                    "SWC_layer_1": get_numeric_array(test_df, "SWC_layer_1"),
                    "SWC_profile_mean": get_numeric_array(test_df, "SWC_profile_mean", fill_nan=True),
                    "m_SM_layer_1": m_layer_1_test,
                    "m_SM_profile_mean": m_profile_test,
                }
            )

            et_export = pd.DataFrame(
                {
                    "TIMESTAMP": test_df["TIMESTAMP"].values,
                    "ET_obs": et_obs_test,
                    "SWC_layer_1": get_numeric_array(test_df, "SWC_layer_1"),
                    "SWC_profile_mean": get_numeric_array(test_df, "SWC_profile_mean", fill_nan=True),
                    "m_SM_layer_1": m_layer_1_test,
                    "m_SM_profile_mean": m_profile_test,
                }
            )

            for model_name, res in no_swc_results.items():
                pred_no_swc = np.asarray(res["pred_test"], dtype=float)
                pred_mswc_layer_1 = pred_no_swc * m_layer_1_test
                pred_mswc_profile = pred_no_swc * m_profile_test

                gs_row.update(flatten_metrics(f"{model_name}_noSWC", safe_metrics(obs_test, pred_no_swc)))
                gs_row.update(
                    flatten_metrics(f"{model_name}_mSWC_layer_1", safe_metrics(obs_test, pred_mswc_layer_1))
                )
                gs_row.update(
                    flatten_metrics(
                        f"{model_name}_mSWC_profile_mean",
                        safe_metrics(obs_test, pred_mswc_profile)
                        if profile_available
                        else (np.nan, np.nan, np.nan, np.nan),
                    )
                )

                gs_export[f"{model_name}_pred_noSWC"] = pred_no_swc
                gs_export[f"{model_name}_pred_mSWC_layer_1"] = pred_mswc_layer_1
                gs_export[f"{model_name}_pred_mSWC_profile_mean"] = pred_mswc_profile

                et_pred_no_swc = estimate_et_pm(test_df, pred_no_swc, scale, gs_target_col="gs_obs")
                et_pred_mswc_layer_1 = estimate_et_pm(
                    test_df, pred_mswc_layer_1, scale, gs_target_col="gs_obs"
                )
                et_pred_mswc_profile = (
                    estimate_et_pm(test_df, pred_mswc_profile, scale, gs_target_col="gs_obs")
                    if profile_available
                    else np.full(len(test_df), np.nan, dtype=float)
                )

                et_row.update(flatten_metrics(f"{model_name}_noSWC", safe_metrics(et_obs_test, et_pred_no_swc)))
                et_row.update(
                    flatten_metrics(f"{model_name}_mSWC_layer_1", safe_metrics(et_obs_test, et_pred_mswc_layer_1))
                )
                et_row.update(
                    flatten_metrics(
                        f"{model_name}_mSWC_profile_mean",
                        safe_metrics(et_obs_test, et_pred_mswc_profile)
                        if profile_available
                        else (np.nan, np.nan, np.nan, np.nan),
                    )
                )

                et_export[f"{model_name}_ET_pred_noSWC"] = et_pred_no_swc
                et_export[f"{model_name}_ET_pred_mSWC_layer_1"] = et_pred_mswc_layer_1
                et_export[f"{model_name}_ET_pred_mSWC_profile_mean"] = et_pred_mswc_profile

            # --------------------------------------------------
            # Framework B: RF directly uses SWC as predictor
            # --------------------------------------------------
            rf_swc_configs = [
                {
                    "model_name": "RF_GPP_VPD_leaf_SWC_layer_1",
                    "feature_cols": ["GPP", "VPD_leaf", "SWC_layer_1"],
                }
            ]

            if profile_available:
                rf_swc_configs.append(
                    {
                        "model_name": "RF_GPP_VPD_leaf_SWC_profile_mean",
                        "feature_cols": ["GPP", "VPD_leaf", "SWC_profile_mean"],
                    }
                )

            for cfg in rf_swc_configs:
                out, err = safe_fit_rf_features(train_df, test_df, cfg["feature_cols"], cfg["model_name"])

                if out is None:
                    gs_row[f"{cfg['model_name']}_status"] = "failed"
                    gs_row[f"{cfg['model_name']}_error_message"] = err
                    et_row[f"{cfg['model_name']}_status"] = "failed"
                    et_row[f"{cfg['model_name']}_error_message"] = err
                    continue

                model_name = out["model_name"]
                pred_test = np.asarray(out["pred_test"], dtype=float)

                gs_row[f"{model_name}_status"] = "success"
                et_row[f"{model_name}_status"] = "success"

                evaluate_and_store_standard_model(
                    model_name=model_name,
                    pred_test=pred_test,
                    obs_test=obs_test,
                    et_obs_test=et_obs_test,
                    test_df=test_df,
                    scale=scale,
                    gs_row=gs_row,
                    et_row=et_row,
                    gs_export=gs_export,
                    et_export=et_export,
                )

            gs_export.to_csv(gs_pred_dir / f"{file_stem}_gs_predictions.csv", index=False)
            et_export.to_csv(et_pred_dir / f"{file_stem}_ET_predictions.csv", index=False)

            pd.DataFrame([gs_row]).to_csv(gs_summary_dir / f"{file_stem}_gs_summary.csv", index=False)
            pd.DataFrame([et_row]).to_csv(et_summary_dir / f"{file_stem}_ET_summary.csv", index=False)

            combined_gs_rows.append(gs_row)
            combined_et_rows.append(et_row)

        except Exception as e:
            err_gs_row = attach_bif_info(
                {
                    "FILE_NAME": file_name,
                    "scale": scale,
                    "status": "failed",
                    "error_message": str(e),
                },
                file_name,
                bif_lookup,
            )
            err_et_row = attach_bif_info(
                {
                    "FILE_NAME": file_name,
                    "scale": scale,
                    "status": "failed",
                    "error_message": str(e),
                },
                file_name,
                bif_lookup,
            )

            combined_gs_rows.append(err_gs_row)
            combined_et_rows.append(err_et_row)

            print(f"[ERROR] {scale} | {file_name} | {e}")
            traceback.print_exc()

    if combined_gs_rows:
        pd.DataFrame(combined_gs_rows).to_csv(gs_summary_dir / f"{scale}_combined_gs_summary.csv", index=False)

    if combined_et_rows:
        pd.DataFrame(combined_et_rows).to_csv(et_summary_dir / f"{scale}_combined_ET_summary.csv", index=False)


def main():
    ensure_dir(BASE_OUTPUT_DIR)
    bif_lookup = load_bif_model_sites(BIF_MODEL_SITES_FILE)

    for scale in SCALES:
        run_scale(scale, bif_lookup)

    print(f"[DONE] Outputs written to: {BASE_OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()