import numpy as np
from utils_metrics import calculate_kge_prime


def calculate_m_sm(swc, n, swc_train):
    """
    Soil moisture stress factor:
      m_SM = 1                                   if SWC > sm_c
      m_SM = (SWC - sm_min) / (sm_c - sm_min)    if SWC <= sm_c

    where sm_c is the nth percentile of training SWC.
    """
    swc = np.asarray(swc, dtype=float)
    swc_train = np.asarray(swc_train, dtype=float)

    swc_train = swc_train[np.isfinite(swc_train)]
    if swc_train.size == 0:
        return np.full_like(swc, np.nan, dtype=float)

    n = float(np.clip(n, 0.0, 100.0))
    sm_min = np.nanmin(swc_train)
    sm_c = np.nanpercentile(swc_train, n)

    if not np.isfinite(sm_min) or not np.isfinite(sm_c) or sm_c <= sm_min:
        return np.ones_like(swc, dtype=float)

    m_sm = np.ones_like(swc, dtype=float)
    mask = swc <= sm_c
    m_sm[mask] = (swc[mask] - sm_min) / (sm_c - sm_min)
    m_sm = np.clip(m_sm, 0.0, 1.0)
    return m_sm


def fit_n_parameter(train_df, pred_train_dict, swc_col, target_col="gs_obs", step=1.0):
    """
    Fit one shared n per site for a given soil moisture variable by maximizing
    mean KGE' across all successful noSWC model predictions on the training period.
    """
    swc_train = train_df[swc_col].to_numpy(dtype=float)
    obs = train_df[target_col].to_numpy(dtype=float)

    swc_valid = np.isfinite(swc_train)
    if swc_valid.sum() < 2:
        return np.nan

    best_n = np.nan
    best_score = -np.inf

    for n in np.arange(0.0, 100.0 + 1e-9, step):
        m_sm = calculate_m_sm(swc_train, n, swc_train)

        scores = []
        for _, pred_train in pred_train_dict.items():
            sim = np.asarray(pred_train, dtype=float) * m_sm
            kge_prime, _, _, _ = calculate_kge_prime(obs, sim)
            if np.isfinite(kge_prime):
                scores.append(kge_prime)

        if not scores:
            continue

        score = float(np.mean(scores))
        if score > best_score:
            best_score = score
            best_n = float(n)

    return best_n
