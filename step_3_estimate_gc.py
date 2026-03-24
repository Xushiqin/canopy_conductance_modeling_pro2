import re
import numpy as np
import pandas as pd
from pathlib import Path

# =========================
# Paths
# =========================
INPUT_BASE_DIR = Path("./fluxnet_extract_fluxmet")
OUTPUT_BASE_DIR = Path("./fluxnet_model_input")

DD_INPUT_DIR = INPUT_BASE_DIR / "DD"
HH_INPUT_DIR = INPUT_BASE_DIR / "HH"

DD_OUTPUT_DIR = OUTPUT_BASE_DIR / "DD"
HH_OUTPUT_DIR = OUTPUT_BASE_DIR / "HH"

DD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Constants
# =========================
pa = 1.293
cp = 0.001013
lamda = 0.0864
LAMBDA_V_J = 2.45e6
MIN_DD_RECORDS = 200

# Thresholds
MIN_VPD = 0.3
MIN_VPD_LEAF = 0.3
MIN_TA = 0.0
MIN_WS = 0.0

# Required SWC layer
REQUIRED_SWC_LAYER = "SWC_F_MDS_1"

# =========================
# Helper functions
# =========================
def get_swc_columns(columns):
    """
    Get all non-QC SWC columns and sort them by layer number.
    Example:
        SWC_F_MDS_1, SWC_F_MDS_2, ...
    """
    swc_cols = [
        c for c in columns
        if ("SWC_F_MDS" in c) and ("QC" not in c)
    ]

    def sort_key(col):
        m = re.search(r"SWC_F_MDS_(\d+)$", col)
        return int(m.group(1)) if m else 9999

    return sorted(swc_cols, key=sort_key)


def extract_site_id(filename):
    parts = filename.split("_")
    if len(parts) < 2:
        return None
    return parts[1].strip()


def build_units_row(columns, time_scale):
    et_unit = "mm day-1" if time_scale == "DD" else "mm hour-1"

    units_map = {
        "TIMESTAMP": "dimensionless",
        "TA": "deg C",
        "Rs": "W m-2",
        "VPD": "kPa",
        "VPD_leaf": "kPa",
        "PA": "kPa",
        "WS": "m s-1",
        "USTAR": "m s-1",
        "SWC_profile_mean": "%",
        "SWC_layer_count": "dimensionless",
        "G": "W m-2",
        "LE": "W m-2",
        "H": "W m-2",
        "Rn": "W m-2",
        "GPP": "g C m-2 d-1",
        "s": "kPa degC-1",
        "gama": "kPa degC-1",
        "ga": "m s-1",
        "ra": "s m-1",
        "gs": "m s-1",
        "rs": "s m-1",
        "ET": et_unit,
    }
    return [units_map.get(col, "dimensionless") for col in columns]


def convert_le_to_et(le_series, time_scale):
    if time_scale == "DD":
        return le_series / LAMBDA_V_J * 86400.0
    elif time_scale == "HH":
        return le_series / LAMBDA_V_J * 3600.0
    else:
        raise ValueError("time_scale must be 'DD' or 'HH'")


# =========================
# Core processing
# =========================
def process_flux_file(file_path, time_scale):
    print(f"Processing: {file_path.name}")

    try:
        df = pd.read_csv(file_path).replace(-9999, np.nan)
    except Exception as e:
        print(f"  Skip: read error -> {e}")
        return None

    required = [
        "TIMESTAMP", "TA_F", "SW_IN_F", "VPD_F", "PA_F", "WS_F", "USTAR",
        "G_F_MDS", "LE_F_MDS", "H_F_MDS", "GPP_NT_VUT_REF"
    ]

    if any(c not in df.columns for c in required):
        print("  Skip: missing required columns.")
        return None

    # =========================
    # SWC
    # =========================
    swc_cols = get_swc_columns(df.columns)

    if len(swc_cols) == 0 or REQUIRED_SWC_LAYER not in swc_cols:
        print(f"  Skip: missing required SWC layer -> {REQUIRED_SWC_LAYER}")
        return None

    data = df[required + swc_cols].copy()

    # Rename SWC columns to unified names
    swc_rename = {
        col: f"SWC_layer_{i+1}"
        for i, col in enumerate(swc_cols)
    }
    data = data.rename(columns=swc_rename)

    swc_layer_cols = list(swc_rename.values())

    # Require at least the first SWC layer to be non-missing at row level
    required_swc_layer_renamed = swc_rename[REQUIRED_SWC_LAYER]
    data = data[data[required_swc_layer_renamed].notna()].copy()

    if data.empty:
        print("  Skip: all rows have missing required SWC values.")
        return None

    # Row-wise valid SWC layer count
    data["SWC_layer_count"] = data[swc_layer_cols].notna().sum(axis=1)

    # Row-wise mean of all available SWC layers
    data["SWC_profile_mean"] = data[swc_layer_cols].mean(axis=1, skipna=True)

    # =========================
    # Rename
    # =========================
    data = data.rename(columns={
        "TA_F": "TA",
        "SW_IN_F": "Rs",
        "VPD_F": "VPD",
        "PA_F": "PA",
        "WS_F": "WS",
        "G_F_MDS": "G",
        "LE_F_MDS": "LE",
        "H_F_MDS": "H",
        "GPP_NT_VUT_REF": "GPP"
    })

    # Convert VPD from hPa to kPa if needed
    data["VPD"] = data["VPD"] * 0.1

    # Derived variables
    data["Rn"] = data["G"] + data["LE"] + data["H"]
    data["ET"] = convert_le_to_et(data["LE"], time_scale)

    # =========================
    # Filtering
    # =========================
    data = data[
        (data["TA"] > MIN_TA) &
        (data["VPD"] > MIN_VPD) &
        (data["G"] > 0) &
        (data["LE"] > 0) &
        (data["H"] > 0) &
        (data["GPP"] > 0) &
        (data["WS"] > MIN_WS)
    ].copy()

    if data.empty:
        print("  Skip: no records after basic filtering.")
        return None

    # Keep only rows with at least one valid SWC value
    data = data[data["SWC_layer_count"] >= 1].copy()

    if data.empty:
        print("  Skip: no records with valid SWC.")
        return None

    # =========================
    # PM inversion
    # =========================
    data["s"] = (
        4098 * 0.6108 * np.exp((17.27 * data["TA"]) / (data["TA"] + 237.3))
    ) / ((data["TA"] + 237.3) ** 2)

    data["gama"] = 0.665e-3 * data["PA"]
    data["ga"] = data["USTAR"] ** 2 / data["WS"]

    data = data[data["ga"] > 0].copy()
    if data.empty:
        print("  Skip: no records with positive ga.")
        return None

    # Use different time conversion for DD and HH
    time_factor = 86400 if time_scale == "DD" else 3600

    denom = (
        data["s"] * (data["Rn"] - data["G"] - data["LE"]) * lamda
        + time_factor * pa * cp * data["VPD"] * data["ga"]
        - data["gama"] * data["LE"] * lamda
    )

    data = data[denom != 0].copy()
    denom = denom.loc[data.index]

    if data.empty:
        print("  Skip: denominator equals zero for all records.")
        return None

    data["gs"] = data["gama"] * data["LE"] * lamda * data["ga"] / denom

    data = data[data["gs"] > 0].copy()
    if data.empty:
        print("  Skip: no positive gs.")
        return None

    data["ra"] = 1.0 / data["ga"]
    data["rs"] = 1.0 / data["gs"]

    data["VPD_leaf"] = (
        (data["gama"] * data["LE"]) / (pa * cp * data["gs"])
    ) * 1e-6

    data = data[data["VPD_leaf"] > MIN_VPD_LEAF].copy()
    if data.empty:
        print("  Skip: no records after VPD_leaf filtering.")
        return None

    # =========================
    # Quantile filter (only gs)
    # =========================
    q01 = data["gs"].quantile(0.01)
    q99 = data["gs"].quantile(0.99)

    data = data[
        (data["gs"] > q01) &
        (data["gs"] <= q99)
    ].copy()

    if data.empty:
        print("  Skip: no records after gs quantile filtering.")
        return None

    # Recalculate rs after gs filtering
    data["rs"] = 1.0 / data["gs"]

    # =========================
    # DD record constraint
    # =========================
    if time_scale == "DD" and len(data) < MIN_DD_RECORDS:
        print(f"  Skip: DD valid records < {MIN_DD_RECORDS}")
        return None

    final_cols = [
        "TIMESTAMP", "TA", "Rs", "VPD", "VPD_leaf", "PA", "WS", "USTAR",
        *swc_layer_cols,
        "SWC_profile_mean",
        "SWC_layer_count",
        "G", "LE", "H", "Rn", "ET", "GPP",
        "s", "gama", "ga", "ra",
        "gs", "rs"
    ]

    return data[final_cols]


# =========================
# Save
# =========================
def save_processed_data(data, output_file, time_scale):
    units_row = build_units_row(data.columns, time_scale)
    combined = pd.concat(
        [pd.DataFrame([units_row], columns=data.columns), data],
        ignore_index=True
    )
    combined.to_csv(output_file, index=False, float_format="%.6f")


# =========================
# Main
# =========================
def main():
    valid_dd_sites = set()

    for f in sorted(DD_INPUT_DIR.glob("*.csv")):
        data = process_flux_file(f, "DD")
        if data is not None:
            save_processed_data(data, DD_OUTPUT_DIR / f.name, "DD")
            site_id = extract_site_id(f.name)
            if site_id:
                valid_dd_sites.add(site_id)

    for f in sorted(HH_INPUT_DIR.glob("*.csv")):
        site_id = extract_site_id(f.name)
        if site_id not in valid_dd_sites:
            continue

        data = process_flux_file(f, "HH")
        if data is not None:
            save_processed_data(data, HH_OUTPUT_DIR / f.name, "HH")

    print("DONE")


if __name__ == "__main__":
    main()