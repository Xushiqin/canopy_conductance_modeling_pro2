import glob
import re
from pathlib import Path

import pandas as pd
import numpy as np
import xarray as xr

# =========================
# Paths
# =========================
INPUT_DIR = Path("./fluxnet/BIF")
OUTPUT_DIR = Path("./fluxnet/BIF_summary")
MODEL_ET_DIR = Path("./fluxnet_model_output/DD/ET_predictions")

DRYNESS_INDEX_FILE = Path(
    "../WettingDryingWorld/output/era5/pet_penman/yearly/dryness_index_yearly.nc"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BIF_SUMMARY_FILE = OUTPUT_DIR / "BIF_summary.csv"
BIF_SUMMARY_IGBP_FILE = OUTPUT_DIR / "BIF_summary_IGBP_count_summary.csv"
BIF_MODEL_SITES_FILE = OUTPUT_DIR / "BIF_model_sites.csv"
BIF_MODEL_SITES_IGBP_FILE = OUTPUT_DIR / "BIF_model_sites_IGBP_count_summary.csv"

# =========================
# Variables
# =========================
TARGET_VARIABLES = [
    "PRODUCT_FIRST_YEAR",
    "PRODUCT_LAST_YEAR",
    "PRODUCT_PROCESSING_CENTER",
    "PRODUCT_SOURCE_NETWORK",
    "HEIGHTC",
    "IGBP",
    "LOCATION_LAT",
    "LOCATION_LONG",
    "MAT",
    "MAP",
    "NETWORK",
]

OUTPUT_COLUMNS = [
    "NUMBER",
    "SITE_ID",
    "FILE_NAME",
    "PRODUCT_FIRST_YEAR",
    "PRODUCT_LAST_YEAR",
    "Temporal_extent",
    "PRODUCT_PROCESSING_CENTER",
    "PRODUCT_SOURCE_NETWORK",
    "HEIGHTC",
    "IGBP",
    "LOCATION_LAT",
    "LOCATION_LONG",
    "MAT",
    "MAP",
    "NETWORK",
    "dryness_index_mean",
]

# =========================
# Helper functions
# =========================
def clean(x):
    return "" if pd.isna(x) else str(x).strip()


def extract_site_id_from_bif_filename(file_name):
    """
    Example:
    FLX_US-Var_BIF_1991-2020_xxx.csv -> US-Var
    Keep the original format from file naming.
    """
    parts = file_name.split("_")
    return parts[1].strip() if len(parts) >= 3 else ""


def extract_site_id_from_et_filename(file_name):
    """
    Example:
    AMF_US-xDJ_FLUXNET_FLUXMET_DD_2019-2024_v1.3_r1.csv -> US-xDJ
    Keep the original format from file naming.
    """
    parts = file_name.split("_")
    return parts[1].strip() if len(parts) >= 2 else ""


def find_value(df, var):
    """
    Extract the first non-empty DATAVALUE for a given VARIABLE.
    Assumes df columns have already been normalized to uppercase.
    """
    if "VARIABLE" not in df.columns or "DATAVALUE" not in df.columns:
        return ""

    matched = df.loc[df["VARIABLE"] == var, "DATAVALUE"]
    for v in matched:
        v = clean(v)
        if v != "":
            return v
    return ""


def to_year(v):
    v = clean(v)
    if v == "":
        return None
    m = re.search(r"\d{4}", v)
    return int(m.group()) if m else None


def save_igbp_count(df, out_file, count_col_name):
    igbp_df = (
        df["IGBP"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
        .value_counts()
        .rename_axis("IGBP")
        .reset_index(name=count_col_name)
    )
    igbp_df.to_csv(out_file, index=False, encoding="utf-8-sig")


# =========================
# Dryness index functions
# =========================
def load_mean_dryness_index(nc_file):
    """
    Read dryness_index_yearly.nc and compute multi-year mean.

    Returns
    -------
    da_mean : xarray.DataArray
        Multi-year mean dryness index.
    lat_name : str
        Latitude coordinate name.
    lon_name : str
        Longitude coordinate name.
    """
    if not nc_file.exists():
        raise FileNotFoundError(f"Dryness index file not found: {nc_file}")

    ds = xr.open_dataset(nc_file)

    if "dryness_index" in ds.data_vars:
        da = ds["dryness_index"]
    elif len(ds.data_vars) == 1:
        da = ds[list(ds.data_vars)[0]]
    else:
        raise ValueError(
            f"Cannot determine dryness index variable automatically. "
            f"Available variables: {list(ds.data_vars)}"
        )

    # Mean over time/year dimension if present
    mean_dims = [d for d in da.dims if d.lower() in ["time", "year"]]
    if mean_dims:
        da_mean = da.mean(dim=mean_dims, skipna=True)
    else:
        da_mean = da

    # Detect lat/lon coordinate names
    lat_name = None
    lon_name = None

    for c in da_mean.coords:
        cl = c.lower()
        if cl in ["lat", "latitude", "y"]:
            lat_name = c
        elif cl in ["lon", "longitude", "x"]:
            lon_name = c

    if lat_name is None or lon_name is None:
        raise ValueError(
            f"Cannot identify latitude/longitude coordinates from: {list(da_mean.coords)}"
        )

    return da_mean, lat_name, lon_name


def extract_dryness_index_for_site(da_mean, lat_name, lon_name, lat, lon):
    """
    Extract nearest-grid dryness index mean for one site.
    """
    try:
        lat = float(lat)
        lon = float(lon)
    except Exception:
        return np.nan

    if np.isnan(lat) or np.isnan(lon):
        return np.nan

    lon_values = da_mean[lon_name].values
    lon_min = np.nanmin(lon_values)
    lon_max = np.nanmax(lon_values)

    # If grid longitude is 0~360 and site longitude is negative
    if lon_min >= 0 and lon_max > 180 and lon < 0:
        lon = lon % 360

    # If grid longitude is -180~180 and site longitude is >180
    elif lon_min < 0 and lon_max <= 180 and lon > 180:
        lon = ((lon + 180) % 360) - 180

    try:
        value = da_mean.sel(
            {lat_name: lat, lon_name: lon},
            method="nearest"
        ).item()
        return value
    except Exception:
        return np.nan


# =========================
# Main
# =========================
def main():
    records = []

    # Load dryness index mean field once
    print(f"[INFO] Loading dryness index: {DRYNESS_INDEX_FILE}")
    da_dry_mean, lat_name, lon_name = load_mean_dryness_index(DRYNESS_INDEX_FILE)
    print(f"[INFO] Dryness index loaded successfully.")
    print(f"[INFO] Coordinates: lat={lat_name}, lon={lon_name}")

    bif_files = sorted(glob.glob(str(INPUT_DIR / "*.csv")))
    if not bif_files:
        print(f"[INFO] No BIF files found in: {INPUT_DIR}")
        return

    for f in bif_files:
        file_name = Path(f).name
        print(f"Processing: {file_name}")

        # Keep SITE_ID exactly as represented in original file naming
        site_id = extract_site_id_from_bif_filename(file_name)

        try:
            df = pd.read_csv(f, dtype=str)
        except Exception as e:
            print(f"[SKIP] Cannot read file: {file_name} | {e}")
            continue

        if df.empty:
            print(f"[SKIP] Empty file: {file_name}")
            continue

        # Normalize column names for searching variables only
        df.columns = [str(c).strip().upper() for c in df.columns]

        if "VARIABLE" in df.columns:
            df["VARIABLE"] = df["VARIABLE"].astype(str).str.strip().str.upper()

        record = {
            "SITE_ID": site_id,
            "FILE_NAME": file_name,
        }

        # Extract variables from VARIABLE / DATAVALUE
        for var in TARGET_VARIABLES:
            record[var] = find_value(df, var)

        # Keep IGBP text cleaned, but do not alter SITE_ID style
        record["IGBP"] = clean(record["IGBP"])

        # Compute temporal extent
        y1 = to_year(record["PRODUCT_FIRST_YEAR"])
        y2 = to_year(record["PRODUCT_LAST_YEAR"])
        record["Temporal_extent"] = y2 - y1 + 1 if y1 is not None and y2 is not None else ""

        # Extract dryness_index_mean using site lat/lon
        lat = pd.to_numeric(clean(record["LOCATION_LAT"]), errors="coerce")
        lon = pd.to_numeric(clean(record["LOCATION_LONG"]), errors="coerce")

        record["dryness_index_mean"] = extract_dryness_index_for_site(
            da_mean=da_dry_mean,
            lat_name=lat_name,
            lon_name=lon_name,
            lat=lat,
            lon=lon,
        )

        records.append(record)

    if not records:
        print("[INFO] No valid records extracted.")
        return

    # =========================
    # Save BIF_summary
    # =========================
    df_all = pd.DataFrame(records)

    for col in OUTPUT_COLUMNS:
        if col != "NUMBER" and col not in df_all.columns:
            df_all[col] = ""

    df_all = df_all[OUTPUT_COLUMNS[1:]]
    df_all = df_all.sort_values("SITE_ID").reset_index(drop=True)
    df_all.insert(0, "NUMBER", range(1, len(df_all) + 1))

    df_all.to_csv(BIF_SUMMARY_FILE, index=False, encoding="utf-8-sig")
    print(f"\n[DONE] Saved: {BIF_SUMMARY_FILE}")

    save_igbp_count(df_all, BIF_SUMMARY_IGBP_FILE, "Total_sites")
    print(f"[DONE] Saved: {BIF_SUMMARY_IGBP_FILE}")

    # =========================
    # Match model sites
    # =========================
    et_files = sorted(glob.glob(str(MODEL_ET_DIR / "*.csv")))
    et_site_ids = {extract_site_id_from_et_filename(Path(f).name) for f in et_files}
    et_site_ids = {x for x in et_site_ids if x != ""}

    print(f"[INFO] ET files found: {len(et_files)}")
    print(f"[INFO] Unique ET SITE_ID found: {len(et_site_ids)}")

    model_df = df_all[df_all["SITE_ID"].isin(et_site_ids)].copy()
    model_df = model_df.sort_values("SITE_ID").reset_index(drop=True)
    model_df["NUMBER"] = range(1, len(model_df) + 1)

    # Ensure same structure as BIF_summary.csv
    model_df = model_df[OUTPUT_COLUMNS]

    model_df.to_csv(BIF_MODEL_SITES_FILE, index=False, encoding="utf-8-sig")
    print(f"[DONE] Saved: {BIF_MODEL_SITES_FILE}")
    print(f"[INFO] Matched model sites: {len(model_df)}")

    save_igbp_count(model_df, BIF_MODEL_SITES_IGBP_FILE, "Site_count")
    print(f"[DONE] Saved: {BIF_MODEL_SITES_IGBP_FILE}")


if __name__ == "__main__":
    main()