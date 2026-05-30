# scorer.py
# Weekly batch scoring pipeline for the Customer Churn Prediction system.
#
# What this does
# --------------
# Once a week, we take the latest customer feature snapshot, run it through
# the trained model, and assign every customer a churn probability and a
# risk tier (HIGH / MEDIUM / LOW). The result is a CSV the retention team
# can act on directly.

import os
from datetime import date

import numpy as np
import pandas as pd


# Folder where weekly scoring CSVs are saved.
OUTPUTS_DIR = "outputs"


def run_weekly_scoring(model, scaler, X_test, feature_names, week_number=1):
    """
    Score a batch of customers, segment them by risk, and save the result.

    Parameters
    ----------
    model : trained sklearn-compatible model
        The model selected by model_trainer.train_and_compare (best_model).
    scaler : sklearn.preprocessing.StandardScaler
        The fitted scaler from preprocessor.preprocess_data. Kept in the
        signature so this function can be reused for fresh, unscaled data
        in the future. We don't apply it here because X_test is already
        scaled by the preprocessor.
    X_test : pandas.DataFrame or numpy.ndarray
        Feature matrix to score (typically the held-out test set, but
        in production this would be the latest customer snapshot).
    feature_names : list
        Feature column names (used to re-wrap arrays as DataFrames).
    week_number : int, default 1
        Which scoring week this run represents. Used in the output
        filename and the scoring_week column.

    Returns
    -------
    pandas.DataFrame
        scoring_df with one row per customer.
    """

    # Make sure outputs/ exists before saving anything.
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Generate churn probability for every customer.
    # ------------------------------------------------------------------
    # predict_proba returns a 2D array: column 0 = P(no churn),
    # column 1 = P(churn). We want column 1.
    # We also defensively scrub any NaN that could break the model,
    # in case the input wasn't fully cleaned upstream.
    if isinstance(X_test, pd.DataFrame):
        X_for_predict = X_test.copy()
    else:
        X_for_predict = pd.DataFrame(X_test, columns=feature_names)

    X_for_predict = X_for_predict.fillna(0)

    churn_probability = model.predict_proba(X_for_predict)[:, 1]

    # ------------------------------------------------------------------
    # 2) Assign risk segments based on probability thresholds.
    # ------------------------------------------------------------------
    #   >= 0.70  -> HIGH    (call now)
    #   0.40 - 0.69 -> MEDIUM (watch list / nurture)
    #   <  0.40  -> LOW     (no action)
    def _risk_level(p):
        if p >= 0.70:
            return "HIGH"
        elif p >= 0.40:
            return "MEDIUM"
        else:
            return "LOW"

    risk_levels = np.array([_risk_level(p) for p in churn_probability])

    # ------------------------------------------------------------------
    # 3) Build the scoring DataFrame.
    # ------------------------------------------------------------------
    # We use the position in X_test as the customer_index. In a real
    # pipeline this would be the actual customerID, but for the test
    # set we keep the row index for traceability.
    today_str = date.today().isoformat()

    scoring_df = pd.DataFrame(
        {
            "customer_index":    np.arange(len(churn_probability)),
            "churn_probability": np.round(churn_probability, 4),
            "risk_level":        risk_levels,
            "scoring_week":      week_number,
            "scoring_date":      today_str,
        }
    )

    # ------------------------------------------------------------------
    # 4) Print a quick summary so the operator sees the segment split.
    # ------------------------------------------------------------------
    total = len(scoring_df)
    high_count   = int((risk_levels == "HIGH").sum())
    medium_count = int((risk_levels == "MEDIUM").sum())
    low_count    = int((risk_levels == "LOW").sum())

    def _pct(n):
        return (n / total) * 100 if total > 0 else 0.0

    print("=" * 70)
    print(f"WEEKLY SCORING SUMMARY (Week {week_number})")
    print("=" * 70)
    print(f"Total customers scored : {total}")
    print(f"HIGH   risk            : {high_count:>6}  ({_pct(high_count):.2f}%)")
    print(f"MEDIUM risk            : {medium_count:>6}  ({_pct(medium_count):.2f}%)")
    print(f"LOW    risk            : {low_count:>6}  ({_pct(low_count):.2f}%)")

    # ------------------------------------------------------------------
    # 5) Save the scoring CSV to outputs/.
    # ------------------------------------------------------------------
    save_path = os.path.join(OUTPUTS_DIR, f"weekly_scoring_week_{week_number}.csv")
    scoring_df.to_csv(save_path, index=False)
    print(f"Saved -> {save_path}\n")

    # ------------------------------------------------------------------
    # 6) Return the DataFrame so downstream code (recommender) can use it.
    # ------------------------------------------------------------------
    return scoring_df


if __name__ == "__main__":
    # Smoke test: full pipeline -> weekly scoring.
    from data_loader import load_and_analyze
    from feature_engineer import engineer_features
    from preprocessor import preprocess_data
    from model_trainer import train_and_compare

    raw_df = load_and_analyze("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df_feat = engineer_features(raw_df)
    X_train, X_test, y_train, y_test, scaler, feature_names, df_processed = preprocess_data(df_feat)

    best_model, all_results, model_name = train_and_compare(
        X_train, X_test, y_train, y_test, feature_names
    )

    scoring_df = run_weekly_scoring(
        best_model, scaler, X_test, feature_names, week_number=1
    )
    print(scoring_df.head())
