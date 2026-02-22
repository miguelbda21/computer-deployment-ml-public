from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


def evaluate_model(model, baseline, X_test, y_test):
    """
    Evaluates model performance using standard regression metrics.
    """

    # --------------------------------------------------
    # 1️⃣ Generate predictions
    # --------------------------------------------------
    predictions = model.predict(X_test)

    # --------------------------------------------------
    # 2️⃣ Compute baseline MAE
    # --------------------------------------------------
    baseline_mae = mean_absolute_error(
        y_test,
        [baseline] * len(y_test)
    )

    # --------------------------------------------------
    # 3️⃣ Compute ML metrics
    # --------------------------------------------------
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    # --------------------------------------------------
    # 4️⃣ Print results
    # --------------------------------------------------
    print("Baseline MAE:", round(baseline_mae, 2))
    print("Model MAE:", round(mae, 2))
    print("Model RMSE:", round(rmse, 2))
    print("Model R²:", round(r2, 3))
