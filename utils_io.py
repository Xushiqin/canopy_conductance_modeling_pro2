from pathlib import Path
import pandas as pd


# =========================
# Required columns
# =========================
REQUIRED_COLUMNS = [
    "TIMESTAMP", "gs", "VPD_leaf", "GPP", "Rs", "TA",
    "SWC_layer_1", "SWC_layer_count"
]


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_site_data(file_path):
    """
    Load one preprocessed site file.
    """

    file_path = Path(file_path)

    try:
        df = pd.read_csv(file_path, skiprows=[1])
    except Exception:
        df = pd.read_csv(file_path)

    # =========================
    # Check required columns
    # =========================
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # =========================
    # Optional column
    # =========================
    if "SWC_profile_mean" not in df.columns:
        df["SWC_profile_mean"] = pd.NA

    keep_cols = REQUIRED_COLUMNS + ["SWC_profile_mean"]
    df = df[keep_cols].copy()

    # =========================
    # 🔥 unify variable name
    # =========================
    df = df.rename(columns={"gs": "gs_obs"})

    # =========================
    # Drop NA
    # =========================
    df = df.dropna(subset=[
        "TIMESTAMP", "gs_obs", "VPD_leaf", "GPP", "Rs", "TA",
        "SWC_layer_1", "SWC_layer_count"
    ])

    # =========================
    # Light filtering
    # =========================
    df = df[
        (df["gs_obs"] > 0) &
        (df["VPD_leaf"] > 0) &
        (df["GPP"] > 0) &
        (df["Rs"] > 0) &
        (df["SWC_layer_1"] > 0)
    ].copy()

    if df.empty:
        raise ValueError("No valid rows remain after filtering.")

    # =========================
    # Ensure numeric
    # =========================
    for col in [
        "gs_obs", "VPD_leaf", "GPP", "Rs", "TA",
        "SWC_layer_1", "SWC_layer_count", "SWC_profile_mean"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # =========================
    # Sort by time
    # =========================
    try:
        ts = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
        if ts.notna().any():
            df = df.assign(_ts=ts).sort_values("_ts").drop(columns="_ts")
    except Exception:
        pass

    return df.reset_index(drop=True)


def split_train_test(df):
    """
    Time-aware split:
      first 50%  -> train
      second 50% -> test
    """
    split_idx = int(len(df) * 0.5)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split resulted in an empty subset.")

    return train_df, test_df