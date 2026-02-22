"""
main.py
Purpose:
- This is the "orchestrator" script.
- It runs the full ML pipeline in the correct order:
  1) Load data from SQL view
  2) Preprocess (time split + encoding)
  3) Train baseline + ML model
  4) Evaluate results
"""

from src.load_data import load_training_data
from src.preprocess import preprocess_data
from src.train_model import train_model
from src.evaluate import evaluate_model


def main():
    # ------------------------------------------------------------
    # 1️⃣ LOAD DATA
    # ------------------------------------------------------------
    df = load_training_data()

    print(f"✅ Loaded rows: {len(df):,}")
    print("✅ Columns:", list(df.columns))

    # ------------------------------------------------------------
    # 2️⃣ PREPROCESS
    # ------------------------------------------------------------
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print(f"✅ Train rows: {len(X_train):,}")
    print(f"✅ Test rows:  {len(X_test):,}")
    print(f"✅ Features after encoding: {X_train.shape[1]:,}")

    # 🔍 DEBUG POINT #1
    print("✅ Preprocessing complete. Starting training now...")

    # ------------------------------------------------------------
    # 3️⃣ TRAIN
    # ------------------------------------------------------------
    model, baseline_value = train_model(X_train, y_train)

    # 🔍 DEBUG POINT #2
    print("✅ Training complete. Starting evaluation now...")

    print(f"✅ Baseline prediction (mean days_to_install): {baseline_value:.2f}")

    # ------------------------------------------------------------
    # 4️⃣ EVALUATE
    # ------------------------------------------------------------
    evaluate_model(model, baseline_value, X_test, y_test)


if __name__ == "__main__":
    main()
