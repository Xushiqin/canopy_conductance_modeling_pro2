import numpy as np


def calculate_kge_prime(obs, sim):
    """
    Compute KGE' and its three components:
      r     : Pearson correlation
      beta  : mean(sim) / mean(obs)
      gamma : CV(sim) / CV(obs)
    """
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)

    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask]
    sim = sim[mask]

    if obs.size < 2:
        return np.nan, np.nan, np.nan, np.nan

    mu_obs = np.mean(obs)
    mu_sim = np.mean(sim)
    sigma_obs = np.std(obs, ddof=0)
    sigma_sim = np.std(sim, ddof=0)

    if mu_obs == 0 or mu_sim == 0 or sigma_obs == 0:
        return np.nan, np.nan, np.nan, np.nan

    r = np.corrcoef(obs, sim)[0, 1]
    beta = mu_sim / mu_obs
    gamma = (sigma_sim / mu_sim) / (sigma_obs / mu_obs)

    if not np.isfinite(r) or not np.isfinite(beta) or not np.isfinite(gamma):
        return np.nan, np.nan, np.nan, np.nan

    kge_prime = 1.0 - np.sqrt((r - 1.0) ** 2 + (beta - 1.0) ** 2 + (gamma - 1.0) ** 2)
    return float(kge_prime), float(r), float(gamma), float(beta)
