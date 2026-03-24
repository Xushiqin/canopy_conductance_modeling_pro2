from sklearn.ensemble import RandomForestRegressor

RF_RANDOM_STATE = 42


def fit_and_predict(train_df, test_df, target_col="gs_obs"):
    X_train = train_df[["GPP", "VPD_leaf"]].copy()
    y_train = train_df[target_col].copy()
    X_test = test_df[["GPP", "VPD_leaf"]].copy()

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RF_RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    return {
        "model_name": "RF_GPP_VPD_leaf",
        "params": {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 2,
            "random_state": RF_RANDOM_STATE,
        },
        "pred_train": model.predict(X_train),
        "pred_test": model.predict(X_test),
    }
