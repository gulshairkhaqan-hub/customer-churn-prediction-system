# model_trainer.py
# Trains and compares three churn prediction models (Logistic Regression,
# XGBoost, LightGBM), evaluates them with the metrics that matter most for
# churn (especially AUC-ROC and Recall), prints a comparison table and
# confusion matrices, picks the best model by AUC-ROC, and saves all
# trained models to the models/ directory using joblib.

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# Folder where serialized models are saved.
MODELS_DIR = "models"


def train_and_compare(X_train, X_test, y_train, y_test, feature_names):
    """
    Train three churn models, compare them, save them, and return the best.

    Parameters
    ----------
    X_train, X_test : pandas.DataFrame
        Scaled feature matrices from preprocessor.preprocess_data().
    y_train, y_test : pandas.Series
        Binary target (1 = churn, 0 = stay).
    feature_names : list
        List of feature column names (kept for consistency with the rest
        of the pipeline; not strictly needed for training itself).

    Returns
    -------
    best_model : trained sklearn-compatible model
        The model with the highest AUC-ROC on the test set.
    all_results : dict
        Dictionary keyed by model name, each value contains the trained
        model and its computed metrics.
    model_name : str
        Name of the best model (e.g., "XGBoost").
    """

    # Make sure the models/ directory exists before we try to save into it.
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Safety net: replace any leftover NaN values with 0.
    # ------------------------------------------------------------------
    # Even after preprocessing, a stray NaN can sneak in (e.g., from a
    # division-by-zero in feature engineering or an unmapped category).
    # Logistic Regression and LightGBM will refuse to train on NaN, so
    # we scrub them here as a last line of defense.
    # We re-wrap the result in a DataFrame so column names survive,
    # which downstream SHAP and visualization code rely on.
    # ------------------------------------------------------------------
    X_train_cols = X_train.columns if hasattr(X_train, "columns") else feature_names
    X_test_cols = X_test.columns if hasattr(X_test, "columns") else feature_names

    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    X_train = pd.DataFrame(X_train, columns=X_train_cols)
    X_test = pd.DataFrame(X_test, columns=X_test_cols)

    # ==================================================================
    # STEP 1 - Define and train the three models
    # ------------------------------------------------------------------
    # We deliberately pick three very different models:
    #   - Logistic Regression : a simple linear baseline
    #   - XGBoost             : strong gradient boosted trees
    #   - LightGBM            : faster gradient boosted trees
    # Each model is configured to handle class imbalance because only
    # ~26.5% of customers churn, so a naive model could just predict
    # "no churn" and look accurate while being useless.
    # ==================================================================
    print("=" * 70)
    print("STEP 1: TRAINING MODELS")
    print("=" * 70)

    # ----- Model 1: Logistic Regression -----
    # Comment: Simple baseline — tells us the minimum performance we should beat.
    # class_weight='balanced' automatically up-weights the minority class.
    # max_iter=1000 gives the optimizer enough room to converge on scaled data.
    print("\n[1/3] Training Logistic Regression (baseline)...")
    lr_model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight="balanced",
    )
    lr_model.fit(X_train, y_train)

    # ----- Model 2: XGBoost -----
    # Comment: scale_pos_weight handles class imbalance (fewer churners
    # than non-churners). It's set to (#negatives / #positives), so the
    # model "feels" the two classes as roughly equal during training.
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos

    print("[2/3] Training XGBoost...")
    xgb_model = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )
    xgb_model.fit(X_train, y_train)

    # ----- Model 3: LightGBM -----
    # Comment: Faster than XGBoost, often similar performance.
    # verbose=-1 keeps LightGBM's training output quiet.
    print("[3/3] Training LightGBM...")
    lgb_model = LGBMClassifier(
        random_state=42,
        class_weight="balanced",
        verbose=-1,
    )
    lgb_model.fit(X_train, y_train)

    # Group everything in one dict so we can iterate cleanly below.
    models = {
        "Logistic Regression": lr_model,
        "XGBoost":             xgb_model,
        "LightGBM":            lgb_model,
    }

    # ==================================================================
    # STEP 2 - Evaluate all three models
    # ------------------------------------------------------------------
    # Why these metrics for churn?
    #   - AUC-ROC  : how well the model RANKS churners above non-churners.
    #                Threshold-independent. Most important overall metric
    #                for churn because we usually score and rank customers.
    #   - Precision: of the customers we flagged as "will churn", how many
    #                actually did? High precision = fewer wasted retention
    #                offers on people who would have stayed anyway.
    #   - Recall   : of the customers who actually churned, how many did
    #                we catch? MOST important for churn — missed churners
    #                are lost revenue. We'd rather over-flag than under-flag.
    #   - F1       : harmonic mean of precision and recall (a balance).
    #   - Accuracy : overall correctness. Misleading on imbalanced data,
    #                so we keep it for reference but don't optimize for it.
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 2: EVALUATING MODELS")
    print("=" * 70)

    all_results = {}

    for name, model in models.items():
        # Hard class predictions (0/1) for precision/recall/F1/accuracy.
        y_pred = model.predict(X_test)
        # Probabilities for the positive class (churn=1) for AUC-ROC.
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "AUC-ROC":   roc_auc_score(y_test, y_proba),
            "Precision": precision_score(y_test, y_pred),
            "Recall":    recall_score(y_test, y_pred),
            "F1":        f1_score(y_test, y_pred),
            "Accuracy":  accuracy_score(y_test, y_pred),
        }

        all_results[name] = {
            "model":   model,
            "metrics": metrics,
            "y_pred":  y_pred,
            "y_proba": y_proba,
        }

    # Build a tidy comparison table using pandas.
    comparison_df = pd.DataFrame(
        {name: result["metrics"] for name, result in all_results.items()}
    ).T  # transpose so rows = models, columns = metrics

    print("\nModel comparison (test set):")
    print(comparison_df.round(4).to_string())

    # ==================================================================
    # STEP 3 - Confusion matrix for each model
    # ------------------------------------------------------------------
    # Confusion matrix layout:
    #
    #                    Predicted: No   Predicted: Yes
    #   Actual: No (0)        TN              FP
    #   Actual: Yes (1)       FN              TP
    #
    #   TP = correctly caught churners (good)
    #   FN = missed churners              (BAD — lost revenue)
    #   FP = false alarms (wasted offer)  (mildly bad)
    #   TN = correctly identified loyal customers
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 3: CONFUSION MATRICES")
    print("=" * 70)

    for name, result in all_results.items():
        cm = confusion_matrix(y_test, result["y_pred"])
        tn, fp, fn, tp = cm.ravel()
        print(f"\n{name}")
        print(f"                 Pred: No   Pred: Yes")
        print(f"  Actual: No   |   {tn:6d}   |   {fp:6d}")
        print(f"  Actual: Yes  |   {fn:6d}   |   {tp:6d}")
        print(f"  -> caught {tp} churners, missed {fn}, "
              f"false alarms {fp}, correctly kept {tn}")

    # ==================================================================
    # STEP 4 - Pick the best model by AUC-ROC
    # ------------------------------------------------------------------
    # AUC-ROC is the right tiebreaker for churn because it measures how
    # well the model RANKS at-risk customers, regardless of threshold.
    # Once we have a good ranking, we can pick any threshold to match
    # the business's appetite for retention spend.
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 4: SELECTING BEST MODEL")
    print("=" * 70)

    model_name = max(
        all_results,
        key=lambda name: all_results[name]["metrics"]["AUC-ROC"],
    )
    best_model = all_results[model_name]["model"]
    best_auc = all_results[model_name]["metrics"]["AUC-ROC"]

    print(f"\nBest model: {model_name} with AUC-ROC: {best_auc:.4f}")

    # ==================================================================
    # STEP 5 - Save all three models + a copy of the best
    # ------------------------------------------------------------------
    # joblib is the standard way to serialize sklearn-compatible models.
    # Saving a separate "best_model.pkl" makes it easy for app.py and
    # main.py to load the chosen model without knowing its name.
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 5: SAVING MODELS")
    print("=" * 70)

    save_paths = {
        "Logistic Regression": os.path.join(MODELS_DIR, "logistic_regression.pkl"),
        "XGBoost":             os.path.join(MODELS_DIR, "xgboost_model.pkl"),
        "LightGBM":            os.path.join(MODELS_DIR, "lightgbm_model.pkl"),
    }

    for name, path in save_paths.items():
        joblib.dump(all_results[name]["model"], path)
        print(f"  Saved {name:<20} -> {path}")

    best_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(best_model, best_path)
    print(f"  Saved BEST model       -> {best_path}")

    print("\nTraining + evaluation complete.\n")

    # ==================================================================
    # STEP 6 - Return artifacts for downstream steps
    # ==================================================================
    return best_model, all_results, model_name


if __name__ == "__main__":
    # Smoke test: run the full data -> features -> preprocess -> train pipeline.
    from data_loader import load_and_analyze
    from feature_engineer import engineer_features
    from preprocessor import preprocess_data

    raw_df = load_and_analyze("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df_feat = engineer_features(raw_df)
    X_train, X_test, y_train, y_test, scaler, feature_names, df_processed = preprocess_data(df_feat)

    best_model, all_results, model_name = train_and_compare(
        X_train, X_test, y_train, y_test, feature_names
    )
    print(f"Final selected model: {model_name}")
