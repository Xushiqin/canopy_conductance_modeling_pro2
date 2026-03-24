import numpy as np
from scipy.optimize import curve_fit


def _medlyn(X, g1):
    vpd, gpp = X
    return g1 * (gpp / np.sqrt(vpd))


def fit_and_predict(train_df, test_df, target_col="gs_obs"):
    x_train = (
        train_df["VPD_leaf"].to_numpy(dtype=float),
        train_df["GPP"].to_numpy(dtype=float),
    )
    y_train = train_df[target_col].to_numpy(dtype=float)

    params, _ = curve_fit(
        _medlyn,
        x_train,
        y_train,
        p0=[0.1],
        bounds=([0.0], [np.inf]),
        maxfev=20000,
    )

    pred_train = _medlyn(x_train, *params)
    x_test = (
        test_df["VPD_leaf"].to_numpy(dtype=float),
        test_df["GPP"].to_numpy(dtype=float),
    )
    pred_test = _medlyn(x_test, *params)

    return {
        "model_name": "Medlyn",
        "params": {
            "g0": 0.0,
            "g1": float(params[0]),
        },
        "pred_train": pred_train,
        "pred_test": pred_test,
    }
