import os
from pathlib import Path
import pandas as pd

# =========================
# Path settings
# =========================
INPUT_BASE_DIR = Path("./fluxnet")
OUTPUT_BASE_DIR = Path("./fluxnet_extract_fluxmet")

# Data folders to process
DATA_TYPES = ["DD", "HH"]

# Create output base directory
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Time column mapping
# =========================
# DD files use 'TIMESTAMP'
# HH files use 'TIMESTAMP_START'
TIME_COLUMN_MAP = {
    "DD": "TIMESTAMP",
    "HH": "TIMESTAMP_START"
}

# =========================
# Required fixed columns
# =========================
REQUIRED_COLUMNS = [
    "TA_F",
    "SW_IN_F",
    "VPD_F",
    "PA_F",
    "WS_F",
    "USTAR",
    "G_F_MDS",
    "LE_F_MDS",
    "H_F_MDS",
    "GPP_NT_VUT_REF"
]

# =========================
# SWC settings
# =========================
# Require at least the first SWC layer
REQUIRED_SWC_COLUMN = "SWC_F_MDS_1"

# Extract all SWC_F_MDS_x columns, but exclude QC columns
SWC_PREFIX = "SWC_F_MDS_"


# =========================
# Processing function
# =========================
def process_fluxnet_folder(data_type):
    """
    Process one Fluxnet folder (DD or HH).

    Rules:
    1. The file must contain the required time column.
    2. The file must contain all required fixed variables.
    3. The file must contain SWC_F_MDS_1.
    4. All available SWC_F_MDS_x columns are extracted.
    5. SWC QC columns (e.g., SWC_F_MDS_1_QC) are excluded.
    6. Output time column is unified as 'TIMESTAMP'.
    """
    input_dir = INPUT_BASE_DIR / data_type
    output_dir = OUTPUT_BASE_DIR / data_type
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    skipped_count = 0

    time_col = TIME_COLUMN_MAP[data_type]
    csv_files = sorted(input_dir.glob("*.csv"))

    print("=" * 70)
    print(f"Processing folder: {input_dir}")
    print(f"Expected time column: {time_col}")
    print(f"Output folder: {output_dir}")
    print("=" * 70)

    if not csv_files:
        print(f"No CSV files found in: {input_dir}")
        return

    for file_path in csv_files:
        try:
            print(f"Processing: {file_path.name}")

            # Read only the header to inspect column names
            df_header = pd.read_csv(file_path, nrows=0)
            all_columns = df_header.columns.tolist()

            # -------------------------
            # Check time column
            # -------------------------
            if time_col not in all_columns:
                print(f"  Skipped: missing time column '{time_col}'")
                skipped_count += 1
                continue

            # -------------------------
            # Check required fixed columns
            # -------------------------
            missing_fixed = [col for col in REQUIRED_COLUMNS if col not in all_columns]

            # -------------------------
            # Find all SWC profile columns
            # Example: SWC_F_MDS_1, SWC_F_MDS_2, ...
            # Exclude QC columns such as SWC_F_MDS_1_QC
            # -------------------------
            swc_columns = [
                col for col in all_columns
                if col.startswith(SWC_PREFIX) and not col.endswith("_QC")
            ]

            # Require at least SWC_F_MDS_1
            has_required_swc = REQUIRED_SWC_COLUMN in swc_columns

            if missing_fixed or not has_required_swc:
                print("  Skipped: missing required variables.")
                if missing_fixed:
                    print(f"    Missing fixed columns: {missing_fixed}")
                if not has_required_swc:
                    print(f"    Missing required SWC column: {REQUIRED_SWC_COLUMN}")
                skipped_count += 1
                continue

            # -------------------------
            # Final columns to extract
            # -------------------------
            columns_to_extract = [time_col] + REQUIRED_COLUMNS + swc_columns

            # Read only selected columns
            df = pd.read_csv(file_path, usecols=columns_to_extract)

            # Rename the time column to a unified name
            df = df.rename(columns={time_col: "TIMESTAMP"})

            # Save extracted file
            output_file = output_dir / file_path.name
            df.to_csv(output_file, index=False)

            print(f"  Saved: {output_file}")
            print(f"  Extracted SWC columns: {swc_columns}")
            processed_count += 1

        except Exception as e:
            print(f"  Error processing {file_path.name}: {e}")
            skipped_count += 1

    print("\nSummary for", data_type)
    print(f"  Successfully extracted: {processed_count}")
    print(f"  Skipped files:          {skipped_count}")
    print(f"  Output directory:       {output_dir}")
    print()


# =========================
# Main
# =========================
def main():
    for data_type in DATA_TYPES:
        process_fluxnet_folder(data_type)


if __name__ == "__main__":
    main()