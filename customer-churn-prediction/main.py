# main.py
# End-to-end pipeline orchestrator for the Customer Churn Prediction system.
#
# Runs every stage in order:
#   1. Load + analyze raw data
#   2. Feature engineering
#   3. Preprocessing (encode + scale + train/test split)
#   4. EDA visualizations
#   5. Train + compare 3 models
#   6. Model comparison + confusion matrix plots
#   7. SHAP explainability
#   8. Weekly batch scoring
#   9. Risk visualizations
#   10. Retention recommendations
#   11. Persist artifacts to models/
#
# Each step is wrapped in try/except with a clear error message so a
# failure in one stage shows up obviously without a giant stack trace.
# When run as a script (python main.py), every step executes in order.

import os
import sys
import traceback

import joblib
import pandas as pd


# ----------------------------------------------------------------------
# Tiny helpers for clean section headers and consistent error reporting.
# ----------------------------------------------------------------------
def _section(step_num, title):
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"STEP {step_num} — {title}")
    print(bar)


def _fatal(step_num, title, err):
    print(f"\n[ERROR] STEP {step_num} ({title}) failed: {err}")
    traceback.print_exc()
    sys.exit(1)


# ----------------------------------------------------------------------
# Map the human-readable model name returned by train_and_compare() to
# the short model_type tag that generate_shap_values() expects.
# ----------------------------------------------------------------------
MODEL_TYPE_MAP = {
    "Logistic Regression": "logistic",
    "XGBoost":             "xgboost",
    "LightGBM":            "lightgbm",
}


def run_pipeline():
    # Make sure the directories the pipeline writes into exist.
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # ==================================================================
    # STEP 1 — Load & Analyze Data
    # ==================================================================
    _section(1, "Load & Analyze Data")
    try:
        from src.data_loader import load_and_analyze
        df = load_and_analyze("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    except Exception as e:
        _fatal(1, "Load & Analyze Data", e)

    # ==================================================================
    # STEP 2 — Feature Engineering
    # ==================================================================
    _section(2, "Feature Engineering")
    try:
        from src.feature_engineer import engineer_features
        df = engineer_features(df)
    except Exception as e:
        _fatal(2, "Feature Engineering", e)

    # ==================================================================
    # STEP 3 — Preprocessing
    # ==================================================================
    _section(3, "Preprocessing")
    try:
        from src.preprocessor import preprocess_data
        (
            X_train, X_test, y_train, y_test,
            scaler, feature_names, df_processed,
        ) = preprocess_data(df)
    except Exception as e:
        _fatal(3, "Preprocessing", e)

    # ==================================================================
    # STEP 4 — EDA Visualizations
    # ------------------------------------------------------------------
    # The EDA plots expect the original, human-readable column values
    # ("Yes"/"No", "Month-to-month", etc.). df_processed has already
    # been encoded, so we run them on the raw `df` returned from
    # data_loader (after feature engineering, before preprocessing).
    # We use the raw df captured into a separate variable so the EDA
    # plots stay readable.
    # ==================================================================
    _section(4, "EDA Visualizations")
    try:
        from src.visualizer import (
            plot_churn_distribution,
            plot_correlation_heatmap,
            plot_churn_by_contract,
            plot_churn_by_tenure,
            plot_monthly_charges_vs_churn,
        )

        # Re-load a fresh raw copy for the EDA plots that need original
        # categorical values (Contract, Churn = Yes/No, etc.).
        eda_df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

        plot_churn_distribution(eda_df)
        plot_correlation_heatmap(eda_df)
        plot_churn_by_contract(eda_df)
        plot_churn_by_tenure(eda_df)
        plot_monthly_charges_vs_churn(eda_df)
    except Exception as e:
        _fatal(4, "EDA Visualizations", e)

    # ==================================================================
    # STEP 5 — Train & Compare Models
    # ==================================================================
    _section(5, "Train & Compare Models")
    try:
        from src.model_trainer import train_and_compare
        best_model, results_dict, best_model_name = train_and_compare(
            X_train, X_test, y_train, y_test, feature_names
        )
    except Exception as e:
        _fatal(5, "Train & Compare Models", e)

    # ==================================================================
    # STEP 6 — Plot Model Comparison + Best-Model Confusion Matrix
    # ==================================================================
    _section(6, "Plot Model Comparison")
    try:
        from src.visualizer import plot_model_comparison, plot_confusion_matrix

        plot_model_comparison(results_dict)

        # Confusion matrix for the best model only.
        best_result = results_dict[best_model_name]
        plot_confusion_matrix(y_test, best_result["y_pred"], best_model_name)
    except Exception as e:
        _fatal(6, "Plot Model Comparison", e)

    # ==================================================================
    # STEP 7 — SHAP Explainability
    # ==================================================================
    _section(7, "SHAP Explainability")
    try:
        from src.explainer import (
            generate_shap_values,
            plot_shap_summary,
            explain_single_customer,
            generate_explanation_text,
        )

        # Translate "XGBoost" / "LightGBM" / "Logistic Regression" to the
        # short tag the explainer wants ("xgboost" / "lightgbm" / "logistic").
        model_type = MODEL_TYPE_MAP.get(best_model_name, best_model_name.lower())

        shap_values, explainer = generate_shap_values(
            best_model, X_train, X_test, feature_names, model_type
        )
        plot_shap_summary(shap_values, X_test, feature_names)

        explanation = explain_single_customer(
            shap_values, X_test, feature_names, customer_index=0
        )
        print("\nExample customer explanation (customer_index=0):")
        print(generate_explanation_text(explanation, feature_names))
    except Exception as e:
        _fatal(7, "SHAP Explainability", e)

    # ==================================================================
    # STEP 8 — Weekly Batch Scoring
    # ==================================================================
    _section(8, "Weekly Batch Scoring")
    try:
        from src.scorer import run_weekly_scoring
        scoring_df = run_weekly_scoring(
            best_model, scaler, X_test, feature_names, week_number=1
        )
    except Exception as e:
        _fatal(8, "Weekly Batch Scoring", e)

    # ==================================================================
    # STEP 9 — Risk Visualization
    # ==================================================================
    _section(9, "Risk Visualization")
    try:
        from src.visualizer import (
            plot_risk_segmentation,
            plot_churn_probability_distribution,
        )
        plot_risk_segmentation(scoring_df)
        plot_churn_probability_distribution(scoring_df)
    except Exception as e:
        _fatal(9, "Risk Visualization", e)

    # ==================================================================
    # STEP 10 — Retention Recommendations
    # ==================================================================
    _section(10, "Retention Recommendations")
    try:
        from src.recommender import generate_recommendations

        # Make sure X_test is a DataFrame with our feature names so the
        # recommender can index into specific customers cleanly.
        if isinstance(X_test, pd.DataFrame):
            X_test_df = X_test.copy()
        else:
            X_test_df = pd.DataFrame(X_test, columns=feature_names)

        recommendations_df = generate_recommendations(
            scoring_df, X_test_df, shap_values, feature_names
        )
    except Exception as e:
        _fatal(10, "Retention Recommendations", e)

    # ==================================================================
    # STEP 11 — Save Everything
    # ==================================================================
    _section(11, "Save Artifacts to models/")
    try:
        joblib.dump(best_model,     "models/best_model.pkl")
        joblib.dump(scaler,         "models/scaler.pkl")
        joblib.dump(results_dict,   "models/results_dict.pkl")
        joblib.dump(feature_names,  "models/feature_names.pkl")
        print("Saved:")
        print("  models/best_model.pkl")
        print("  models/scaler.pkl")
        print("  models/results_dict.pkl")
        print("  models/feature_names.pkl")
    except Exception as e:
        _fatal(11, "Save Artifacts", e)

    print("\n" + "=" * 70)
    print("Pipeline complete! Run: streamlit run app.py")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
