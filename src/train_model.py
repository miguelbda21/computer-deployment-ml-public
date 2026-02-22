from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def train_model(X_train, y_train):
    """
    Trains a RandomForest regression model.

    Why RandomForest:
    - Handles non-linear patterns
    - Works well with tabular business data
    - Requires minimal tuning
    """

    # --------------------------------------------------
    # 1️⃣ Baseline prediction
    # --------------------------------------------------
    # Predicting the mean of y_train
    baseline_prediction = y_train.mean()

    # --------------------------------------------------
    # 2️⃣ Train RandomForest model
    # --------------------------------------------------
    model = RandomForestRegressor(
        n_estimators=200,       # number of trees
        max_depth=10,           # limits overfitting
        random_state=42
    )

    model.fit(X_train, y_train)

    return model, baseline_prediction
