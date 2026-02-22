# src/preprocess.py

import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def preprocess_data(df: pd.DataFrame):
    """
    OPTION A (chosen): DROP dv_department

    This preprocessing pipeline:
    - Time-splits data using sys_created_on (prevents leakage)
    - Creates X/y for train + test
    - Drops ID columns (sys_id, dv_model_id, dv_model_number)
    - Drops dv_department (Option A)
    - Keeps dv_location signal via FREQUENCY encoding (1 numeric column)
    - One-hot encodes only a small, safe set of categorical columns:
        device_type, dv_manufacturer, dv_cpu_type, dv_install_status
    - Ensures output is fully numeric (required by RandomForest)
    """

    # 1) Sort by time for leakage-safe split
    if "sys_created_on" not in df.columns:
        raise ValueError("Expected column 'sys_created_on' but it was not found.")
    df = df.sort_values("sys_created_on").copy()

    # 2) Time-based split (80/20)
    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    # 3) Separate target
    if "days_to_install" not in df.columns:
        raise ValueError("Expected column 'days_to_install' but it was not found.")
    y_train = train_df["days_to_install"]
    y_test = test_df["days_to_install"]

    # Build feature matrices (drop target + split column)
    X_train = train_df.drop(columns=["days_to_install", "sys_created_on"], errors="ignore")
    X_test = test_df.drop(columns=["days_to_install", "sys_created_on"], errors="ignore")

    # 4) Drop identifier columns (IDs are labels, not predictive features)
    id_cols = ["sys_id", "dv_model_id", "dv_model_number"]
    X_train = X_train.drop(columns=[c for c in id_cols if c in X_train.columns])
    X_test = X_test.drop(columns=[c for c in id_cols if c in X_test.columns])

    # ✅ Option A: drop dv_department (prevents high-cardinality one-hot explosion)
    X_train = X_train.drop(columns=["dv_department"], errors="ignore")
    X_test = X_test.drop(columns=["dv_department"], errors="ignore")

    # 5) Keep dv_location but avoid one-hot explosion (frequency encoding)
    if "dv_location" in X_train.columns:
        loc_freq = X_train["dv_location"].value_counts(normalize=True)
        X_train["dv_location_freq"] = X_train["dv_location"].map(loc_freq)
        X_test["dv_location_freq"] = X_test["dv_location"].map(loc_freq).fillna(0.0)

        X_train = X_train.drop(columns=["dv_location"])
        X_test = X_test.drop(columns=["dv_location"])

    # 6) Ensure dv_ram is numeric (handles values like "16 GB")
    if "dv_ram" in X_train.columns:
        X_train["dv_ram"] = pd.to_numeric(
            X_train["dv_ram"].astype(str).str.replace(r"\D", "", regex=True),
            errors="coerce"
        )
        X_test["dv_ram"] = pd.to_numeric(
            X_test["dv_ram"].astype(str).str.replace(r"\D", "", regex=True),
            errors="coerce"
        )

    # 7) One-hot encode only selected categorical columns (keeps feature count small)
    keep_onehot = ["device_type", "dv_manufacturer", "dv_cpu_type", "dv_install_status"]

    categorical_cols = X_train.select_dtypes(include="object").columns.tolist()
    categorical_cols = [c for c in categorical_cols if c in keep_onehot]

    # Version-safe OneHotEncoder
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
    X_test_encoded = encoder.transform(X_test[categorical_cols])

    encoded_cols = encoder.get_feature_names_out(categorical_cols)

    X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoded_cols, index=X_train.index)
    X_test_encoded = pd.DataFrame(X_test_encoded, columns=encoded_cols, index=X_test.index)

    # Drop original categorical cols then append encoded cols
    X_train = X_train.drop(columns=categorical_cols, errors="ignore")
    X_test = X_test.drop(columns=categorical_cols, errors="ignore")

    X_train = pd.concat([X_train, X_train_encoded], axis=1)
    X_test = pd.concat([X_test, X_test_encoded], axis=1)

    # 8) Fill missing values (RandomForest cannot handle NaN)
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # 9) Final safety check: ensure everything is numeric
    non_numeric_cols = X_train.select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric_cols:
        raise ValueError(
            f"Non-numeric columns remain after preprocessing: {non_numeric_cols}. "
            f"Drop or encode them before training."
        )

    return X_train, X_test, y_train, y_test
