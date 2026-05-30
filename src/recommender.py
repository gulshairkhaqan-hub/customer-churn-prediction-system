# recommender.py
# Rule-based retention recommendation engine for HIGH-risk customers.
#
# Workflow
# --------
# 1. Take the scoring_df produced by scorer.run_weekly_scoring().
# 2. For every HIGH-risk customer, look at the top SHAP features that drove
#    their churn probability up.
# 3. Map the top reason to a concrete retention action using simple,
#    business-readable rules. A retention analyst can read the CSV and
#    act on it without having to understand SHAP at all.

import os

import numpy as np
import pandas as pd


# Folder where the recommendations CSV is saved.
OUTPUTS_DIR = "outputs"


# ----------------------------------------------------------------------
# Internal helper: normalize SHAP output across versions / model types.
# ----------------------------------------------------------------------
# Mirrors the helper in explainer.py so this module can stand on its own.
# We always want a 2D (n_samples, n_features) array for the positive
# class (churn = 1).
def _to_positive_class_array(shap_values):
    if hasattr(shap_values, "values"):
        shap_values = shap_values.values

    if isinstance(shap_values, (list, tuple)):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        if shap_values.shape[-1] > 1:
            shap_values = shap_values[:, :, 1]
        else:
            shap_values = shap_values[:, :, 0]

    return shap_values


# ----------------------------------------------------------------------
# Internal helper: pick a retention action based on the top SHAP feature.
# ----------------------------------------------------------------------
# Rules are checked in priority order. The first match wins.
# We use case-insensitive substring matches so the rules work whether
# the column is "Contract", "contract_risk", "MonthlyCharges", etc.
def _recommend_action(top_feature):
    f = str(top_feature).lower()

    # Month-to-month / contract risk -> push customer onto a longer term.
    if "contract" in f:
        return "Offer annual contract with 20% discount"

    # Tenure (low) -> assign a CSM to build the relationship early.
    # NOTE: checked BEFORE the charges rule because "TotalCharges"
    # doesn't contain "tenure"; we only want true tenure features here.
    if "tenure" in f:
        return "Assign dedicated customer success manager"

    # Pricing concerns -> retention discount or loyalty perks.
    if "monthlycharges" in f or "totalcharges" in f or "charges_per_month_ratio" in f:
        return "Offer discounted plan or loyalty reward"

    # Lack of support services -> remove the friction with a free upgrade.
    if "techsupport" in f or "onlinesecurity" in f or "no_support_services" in f:
        return "Offer free tech support upgrade for 3 months"

    # Fiber complaints are a known churn driver in this dataset.
    if "internetservice" in f or "fiber" in f:
        return "Investigate service quality complaints"

    # Catch-all: get a human on the phone.
    return "Schedule retention call within 48 hours"


def generate_recommendations(scoring_df, X_test_df, shap_values, feature_names):
    """
    Generate retention recommendations for every HIGH-risk customer.

    Parameters
    ----------
    scoring_df : pandas.DataFrame
        The output of scorer.run_weekly_scoring(), containing
        customer_index, churn_probability, risk_level, etc.
    X_test_df : pandas.DataFrame or numpy.ndarray
        The same feature matrix used for scoring. Used here only to make
        sure customer indexing lines up with shap_values.
    shap_values : numpy.ndarray or shap.Explanation
        SHAP values for the same customers as scoring_df, produced by
        explainer.generate_shap_values().
    feature_names : list
        Feature column names corresponding to shap_values columns.

    Returns
    -------
    pandas.DataFrame
        recommendations_df with one row per HIGH-risk customer.
    """

    # Make sure outputs/ exists before saving the CSV.
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # Normalize SHAP values to a clean 2D positive-class array.
    shap_values = _to_positive_class_array(shap_values)

    # ------------------------------------------------------------------
    # Step 1: select HIGH-risk customers only.
    # ------------------------------------------------------------------
    high_risk = scoring_df[scoring_df["risk_level"] == "HIGH"].copy()

    # If nothing is HIGH risk this week, return an empty frame with the
    # right columns so downstream code doesn't blow up.
    if len(high_risk) == 0:
        print("No HIGH-risk customers this week. No recommendations generated.\n")
        empty_df = pd.DataFrame(
            columns=[
                "customer_index",
                "risk_level",
                "churn_probability",
                "top_reason_1",
                "top_reason_2",
                "top_reason_3",
                "recommended_action",
            ]
        )
        save_path = os.path.join(OUTPUTS_DIR, "retention_recommendations.csv")
        empty_df.to_csv(save_path, index=False)
        return empty_df

    # ------------------------------------------------------------------
    # Step 2: for each HIGH-risk customer, pull their top 3 SHAP features.
    # ------------------------------------------------------------------
    # We want the features that pushed risk UP the most, so we sort by
    # SHAP value descending (large positive value = strongest push toward
    # churn). If no features have positive SHAP for a customer (rare for
    # HIGH-risk), we fall back to absolute magnitude so we still surface
    # the strongest drivers.
    feature_names = list(feature_names)
    rows = []

    for _, row in high_risk.iterrows():
        idx = int(row["customer_index"])
        customer_shap = shap_values[idx]

        order = np.argsort(-customer_shap)  # descending: most positive first
        top_idx = order[:3]

        # Safety: if all top values are non-positive, use magnitude.
        if customer_shap[top_idx[0]] <= 0:
            order = np.argsort(-np.abs(customer_shap))
            top_idx = order[:3]

        top_features = [feature_names[i] for i in top_idx]

        # Map the #1 reason to a recommended action.
        action = _recommend_action(top_features[0])

        # Pad to length 3 just in case (shouldn't happen with real data).
        while len(top_features) < 3:
            top_features.append("")

        rows.append(
            {
                "customer_index":     idx,
                "risk_level":         row["risk_level"],
                "churn_probability":  row["churn_probability"],
                "top_reason_1":       top_features[0],
                "top_reason_2":       top_features[1],
                "top_reason_3":       top_features[2],
                "recommended_action": action,
            }
        )

    recommendations_df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Step 3: save and preview.
    # ------------------------------------------------------------------
    save_path = os.path.join(OUTPUTS_DIR, "retention_recommendations.csv")
    recommendations_df.to_csv(save_path, index=False)

    print("=" * 70)
    print("RETENTION RECOMMENDATIONS")
    print("=" * 70)
    print(f"HIGH-risk customers           : {len(recommendations_df)}")
    print(f"Saved recommendations CSV     : {save_path}\n")
    print("First 10 recommendations:")
    print(recommendations_df.head(10).to_string(index=False))
    print()

    return recommendations_df


if __name__ == "__main__":
    # Smoke test: full pipeline -> SHAP -> scoring -> recommendations.
    from data_loader import load_and_analyze
    from feature_engineer import engineer_features
    from preprocessor import preprocess_data
    from model_trainer import train_and_compare
    from explainer import generate_shap_values
    from scorer import run_weekly_scoring

    raw_df = load_and_analyze("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df_feat = engineer_features(raw_df)
    X_train, X_test, y_train, y_test, scaler, feature_names, df_processed = preprocess_data(df_feat)

    best_model, all_results, model_name = train_and_compare(
        X_train, X_test, y_train, y_test, feature_names
    )

    model_type_map = {
        "Logistic Regression": "logistic",
        "XGBoost":             "xgboost",
        "LightGBM":            "lightgbm",
    }
    model_type = model_type_map[model_name]

    shap_values, _ = generate_shap_values(
        best_model, X_train, X_test, feature_names, model_type
    )

    scoring_df = run_weekly_scoring(
        best_model, scaler, X_test, feature_names, week_number=1
    )

    recommendations_df = generate_recommendations(
        scoring_df, X_test, shap_values, feature_names
    )
    print("Total recommendations generated:", len(recommendations_df))
