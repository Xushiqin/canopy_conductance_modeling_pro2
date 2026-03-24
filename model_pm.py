import numpy as np
import pandas as pd

RHO_AIR = 1.293       # kg m-3
CP_AIR = 1004.67      # J kg-1 K-1
LAMBDA_V = 2.45e6     # J kg-1


def get_timestep_seconds(scale):
    """Return the number of seconds for each time step."""
    if scale == "DD":
        return 86400.0
    elif scale == "HH":
        return 3600.0
    else:
        raise ValueError(f"Unsupported scale: {scale}")


def get_observed_et(df, scale):
    """
    Get observed ET from processed input data.

    Priority:
    1. Use ET directly if available.
    2. Otherwise convert LE to ET.
    """
    if "ET" in df.columns:
        return pd.to_numeric(df["ET"], errors="coerce").to_numpy(dtype=float)

    if "LE" in df.columns:
        seconds = get_timestep_seconds(scale)
        le = pd.to_numeric(df["LE"], errors="coerce").to_numpy(dtype=float)
        return le / LAMBDA_V * seconds

    return np.full(len(df), np.nan)


def convert_predicted_gs_to_m_s(df, pred_gs, target_col="gs_obs"):
    """
    Convert predicted gs to m s-1 using observed scaling.

    If both target_col and gs_m_s-1 exist in df, use:
        factor = gs_m_s-1 / target_col

    Otherwise assume pred_gs is already in m s-1.
    """
    pred_gs = np.asarray(pred_gs, dtype=float)

    if ("gs_m_s-1" not in df.columns) or (target_col not in df.columns):
        return pred_gs.copy()

    obs_target = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)
    obs_gs_m_s = pd.to_numeric(df["gs_m_s-1"], errors="coerce").to_numpy(dtype=float)

    factor = np.divide(
        obs_gs_m_s,
        obs_target,
        out=np.full(len(df), np.nan, dtype=float),
        where=np.isfinite(obs_target) & (obs_target != 0)
    )

    return pred_gs * factor


def estimate_et_pm(df, gs_pred, scale, gs_target_col="gs_obs"):
    """
    Estimate ET using the Penman-Monteith equation.

    Required columns in df:
    - s     : slope of saturation vapor pressure curve (kPa °C-1)
    - gama  : psychrometric constant (kPa °C-1)
    - Rn    : net radiation (W m-2)
    - G     : soil heat flux (W m-2)
    - VPD   : vapor pressure deficit (kPa)
    - ra    : aerodynamic resistance (s m-1)

    Parameters
    ----------
    df : pandas.DataFrame
    gs_pred : array-like
        Predicted gs in the same space as gs_target_col.
    scale : str
        "DD" or "HH"
    gs_target_col : str
        Usually "gs_obs"

    Returns
    -------
    numpy.ndarray
        ET in mm day-1 (DD) or mm hour-1 (HH)
    """
    seconds = get_timestep_seconds(scale)

    s = pd.to_numeric(df["s"], errors="coerce").to_numpy(dtype=float)
    gamma = pd.to_numeric(df["gama"], errors="coerce").to_numpy(dtype=float)
    rn = pd.to_numeric(df["Rn"], errors="coerce").to_numpy(dtype=float)
    g = pd.to_numeric(df["G"], errors="coerce").to_numpy(dtype=float)
    vpd = pd.to_numeric(df["VPD"], errors="coerce").to_numpy(dtype=float)
    ra = pd.to_numeric(df["ra"], errors="coerce").to_numpy(dtype=float)

    gs_m_s = convert_predicted_gs_to_m_s(df, gs_pred, target_col=gs_target_col)

    valid_gs = np.isfinite(gs_m_s) & (gs_m_s > 0)
    rs = np.full(len(df), np.nan, dtype=float)
    rs[valid_gs] = 1.0 / gs_m_s[valid_gs]

    numerator = s * (rn - g) + (RHO_AIR * CP_AIR * vpd / ra)
    denominator = s + gamma * (1.0 + rs / ra)

    lambda_e = np.divide(
        numerator,
        denominator,
        out=np.full(len(df), np.nan, dtype=float),
        where=np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
    )

    et = lambda_e / LAMBDA_V * seconds
    et[~np.isfinite(et)] = np.nan

    return et