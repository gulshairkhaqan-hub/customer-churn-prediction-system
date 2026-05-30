# explainer.py
# SHAP-based model explainability for the Customer Churn Prediction system.
#
# What is SHAP?
# -------------
# SHAP (SHapley Additive exPlanations) tells us, for each prediction, how
# much each feature pushed the model's output up or down.
#   - Positive SHAP value -> feature INCREASED predicted churn risk
#   - Negative SHAP value -> feature DECREASED predicted churn risk
# Averaging |SHAP| across all customers gives us global feature importance:
# the features that matter most for churn overall.

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


# Folder where SHAP plots are saved.
OUTPUTS_DIR = "outputs"


# ----------------------------------------------------------------------
# Internal helper: normalize SHAP output across versions / model types.
# ----------------------------------------------------------------------
# Different SHAP versions (and different models) return SHAP values in
# slightly different shapes:
#   - a plain 2D numpy array of shape (n_samples, n_features)
#   - a list of two arrays [class_0_values, class_1_values] (older API)
#   - a 3D numpy array of shape (n_samples, n_features, n_classes)
#   - a shap.Explanation object with a .values attribute
# We always want a 2D (n_samples, n_features) array for the POSITIVE
# class (churn = 1). This helper unwraps whatever SHAP returned.
def _to_positive_class_array(shap_values):
    # shap.Explanation object -> use its .values attribute.
    if hasattr(shap_values, "values"):
        shap_values = shap_values.values

    # List/tuple form: [class_0_array, class_1_array]
    if isinstance(shap_values, (list, tuple)):
        # Pick class 1 (churn) if it exists, otherwise the only entry.
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    shap_values = np.asarray(shap_values)

    # 3D array: (n_samples, n_features, n_classes) -> take class 1.
    if shap_values.ndim == 3:
        if shap_values.shape[-1] > 1:
            shap_values = shap_values[:, :, 1]
        else:
            shap_values = shap_values[:, :, 0]

    return shap_values


# ======================================================================
# FUNCTION 1: generate_shap_values
# ======================================================================
def generate_shap_values(model, X_train, X_test, feature_names, model_type):
    """
    Build a SHAP explainer for the given model and compute SHAP values
    for every row in X_test.

    Parameters
    ----------
    model : trained sklearn-compatible model
        Logistic Regression, XGBoost, or LightGBM model.
    X_train : pandas.DataFrame or numpy.ndarray
        Training features. Used as background data for LinearExplainer.
    X_test : pandas.DataFrame or numpy.ndarray
        Features to explain (we explain the test set).
    feature_names : list
        List of feature column names.
    model_type : str
        One of {"xgboost", "lightgbm", "logistic"}.

    Returns
    -------
    shap_values : numpy.ndarray
        2D array of shape (n_test_samples, n_features) with SHAP values
        for the positive class (churn = 1).
    explainer : shap.Explainer
        The fitted SHAP explainer (kept in case downstream code needs it).
    """

    print("Building SHAP explainer for model_type =", model_type)

    model_type = model_type.lower()

    # We try the "right" explainer for each model family, but fall back
    # to the more general shap.Explainer if anything goes wrong (SHAP's
    # API has changed across versions).
    try:
        if model_type in ("xgboost", "lightgbm"):
            # TreeExplainer is exact and very fast for tree-based models.
            explainer = shap.TreeExplainer(model)
            raw_shap = explainer.shap_values(X_test)

        elif model_type == "logistic":
            # LinearExplainer needs background data so it knows the
            # "average" customer to compare against.
            explainer = shap.LinearExplainer(model, X_train)
            raw_shap = explainer.shap_values(X_test)

        else:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                "Use 'xgboost', 'lightgbm', or 'logistic'."
            )

    except Exception as e:
        # Fallback: shap.Explainer auto-selects an appropriate algorithm.
        print(f"  Primary SHAP explainer failed ({e}). Falling back to shap.Explainer...")
        explainer = shap.Explainer(model, X_train)
        raw_shap = explainer(X_test)

    # Normalize to a 2D positive-class array, regardless of the SHAP
    # version or model output shape.
    shap_values = _to_positive_class_array(raw_shap)

    print(f"  SHAP values shape: {shap_values.shape}")
    return shap_values, explainer


# ======================================================================
# FUNCTION 2: plot_shap_summary
# ======================================================================
def plot_shap_summary(shap_values, X_test, feature_names):
    """
    Create a SHAP summary bar plot showing the top 15 features by
    average absolute SHAP value (i.e., the top 15 churn drivers).

    This shows WHICH features matter most for churn overall, across
    all test customers.

    Parameters
    ----------
    shap_values : numpy.ndarray
        2D SHAP value array (n_samples, n_features).
    X_test : pandas.DataFrame or numpy.ndarray
        Test features (used so SHAP can show feature values too).
    feature_names : list
        Feature column names (used as labels on the chart).

    Returns
    -------
    str
        Path to the saved image file.
    """

    # Make sure outputs/ exists before saving.
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUTS_DIR, "shap_summary.png")

    # Make sure SHAP values and X_test are aligned shape-wise.
    shap_values = _to_positive_class_array(shap_values)

    # Convert X_test to a DataFrame with the right column names so SHAP
    # uses our feature names on the plot.
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=feature_names)

    try:
        # SHAP's built-in summary plot. plot_type="bar" shows mean(|SHAP|).
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X_test,
            feature_names=feature_names,
            plot_type="bar",
            max_display=15,
            show=False,
        )
        plt.title("Top 15 Churn Drivers (SHAP Feature Importance)")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    except Exception as e:
        # Fallback: build the bar chart manually from mean(|SHAP|).
        print(f"  shap.summary_plot failed ({e}). Falling back to manual bar plot.")
        mean_abs = np.abs(shap_values).mean(axis=0)
        importance = (
            pd.DataFrame({"feature": feature_names, "importance": mean_abs})
            .sort_values("importance", ascending=False)
            .head(15)
        )

        plt.figure(figsize=(10, 8))
        plt.barh(importance["feature"][::-1], importance["importance"][::-1])
        plt.xlabel("mean(|SHAP value|)")
        plt.title("Top 15 Churn Drivers (SHAP Feature Importance)")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  Saved SHAP summary plot -> {save_path}")
    return save_path


# ======================================================================
# FUNCTION 3: explain_single_customer
# ======================================================================
def explain_single_customer(shap_values, X_test, feature_names, customer_index=0):
    """
    Explain ONE customer's churn prediction by surfacing the top 3
    features that pushed their risk up or down the most.

    Parameters
    ----------
    shap_values : numpy.ndarray
        2D SHAP array (n_samples, n_features).
    X_test : pandas.DataFrame or numpy.ndarray
        Test features (kept for consistency / future use).
    feature_names : list
        Feature column names.
    customer_index : int, default 0
        Which row of X_test to explain.

    Returns
    -------
    dict
        {
          "top_reasons":    [feature_name_1, feature_name_2, feature_name_3],
          "shap_values":    [shap_val_1, shap_val_2, shap_val_3],
          "risk_direction": ["increases risk" | "decreases risk", ...],
        }
    """

    shap_values = _to_positive_class_array(shap_values)

    # Pull out the SHAP values for the customer we want to explain.
    customer_shap = shap_values[customer_index]

    # Build a tidy DataFrame: feature | shap_value | direction.
    # Direction tells the analyst whether this feature pushed the
    # prediction toward churn ("increases risk") or away from it.
    explanation_df = pd.DataFrame(
        {
            "feature":    feature_names,
            "shap_value": customer_shap,
        }
    )
    explanation_df["direction"] = np.where(
        explanation_df["shap_value"] > 0,
        "increases risk",
        "decreases risk",
    )

    # Sort by absolute SHAP value (biggest impact first), and grab the
    # top 3 most influential features for this customer.
    explanation_df["abs_shap"] = explanation_df["shap_value"].abs()
    top3 = explanation_df.sort_values("abs_shap", ascending=False).head(3)

    return {
        "top_reasons":    top3["feature"].tolist(),
        "shap_values":    top3["shap_value"].tolist(),
        "risk_direction": top3["direction"].tolist(),
    }


# ======================================================================
# FUNCTION 4: generate_explanation_text
# ======================================================================
def generate_explanation_text(explanation_dict, feature_names):
    """
    Convert the structured explanation dict into a human-readable
    paragraph that a non-technical user (e.g., a retention analyst)
    can read at a glance.

    Parameters
    ----------
    explanation_dict : dict
        The dict returned by explain_single_customer().
    feature_names : list
        Feature column names (kept for signature parity / future use).

    Returns
    -------
    str
        A multi-line, numbered explanation string.
    """

    # Map the technical column names produced by preprocessing/feature
    # engineering to friendly business phrases the reader will recognize.
    # Anything not in the map falls through to a sensible auto-generated
    # phrase based on the column name.
    friendly_names = {
        "tenure":                              "Short tenure",
        "tenure_group":                        "New customer lifecycle stage",
        "MonthlyCharges":                      "High monthly charges",
        "TotalCharges":                        "Total charges to date",
        "charges_per_month_ratio":             "Recent billing spike",
        "has_multiple_services":               "Few add-on services",
        "is_high_value":                       "High-value customer",
        "contract_risk":                       "Month-to-month contract",
        "payment_risk":                        "Electronic-check payment",
        "no_support_services":                 "No support services",
        "PaperlessBilling":                    "Paperless billing",
        "SeniorCitizen":                       "Senior citizen status",
        "Partner":                             "No partner",
        "Dependents":                          "No dependents",
        "PhoneService":                        "Phone service status",
        "InternetService_Fiber optic":         "Fiber optic internet",
        "InternetService_No":                  "No internet service",
        "OnlineSecurity_No internet service":  "No online security (no internet)",
        "OnlineSecurity_Yes":                  "Has online security",
        "OnlineBackup_No internet service":    "No online backup (no internet)",
        "OnlineBackup_Yes":                    "Has online backup",
        "DeviceProtection_No internet service": "No device protection (no internet)",
        "DeviceProtection_Yes":                "Has device protection",
        "TechSupport_No internet service":     "No tech support (no internet)",
        "TechSupport_Yes":                     "Has tech support",
        "StreamingTV_No internet service":     "No streaming TV (no internet)",
        "StreamingTV_Yes":                     "Has streaming TV",
        "StreamingMovies_No internet service": "No streaming movies (no internet)",
        "StreamingMovies_Yes":                 "Has streaming movies",
        "MultipleLines_No phone service":      "No phone service",
        "MultipleLines_Yes":                   "Multiple phone lines",
    }

    def _pretty(name):
        if name in friendly_names:
            return friendly_names[name]
        # Fallback: replace underscores with spaces and capitalize.
        return name.replace("_", " ").capitalize()

    top_reasons    = explanation_dict.get("top_reasons", [])
    risk_direction = explanation_dict.get("risk_direction", [])

    # Decide the headline sentence based on which way the top features push.
    # If most of them push risk UP, the customer is at risk; otherwise loyal.
    increases = sum(1 for d in risk_direction if d == "increases risk")
    if increases >= len(risk_direction) / 2:
        header = "This customer is at risk because:"
    else:
        header = "This customer is likely to stay because:"

    # Build a numbered list of reasons, one per top feature.
    lines = [header]
    for i, (feat, direction) in enumerate(zip(top_reasons, risk_direction), start=1):
        # Shape the verb based on direction and pluralization sounds natural.
        verb = "increases" if direction == "increases risk" else "decreases"
        lines.append(f"{i}. {_pretty(feat)} {verb} churn risk")

    return "\n".join(lines)


if __name__ == "__main__":
    # Smoke test: full pipeline -> SHAP -> summary plot -> single-customer
    # explanation -> human-readable text.
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

    # Map the human model name back to the type tag SHAP needs.
    model_type_map = {
        "Logistic Regression": "logistic",
        "XGBoost":             "xgboost",
        "LightGBM":             "lightgbm",
    }
    model_type = model_type_map[model_name]

    shap_values, explainer = generate_shap_values(
        best_model, X_train, X_test, feature_names, model_type
    )
    plot_shap_summary(shap_values, X_test, feature_names)

    explanation = explain_single_customer(shap_values, X_test, feature_names, customer_index=0)
    print("\nSingle-customer explanation dict:")
    print(explanation)

    print("\nHuman-readable explanation:")
    print(generate_explanation_text(explanation, feature_names))
