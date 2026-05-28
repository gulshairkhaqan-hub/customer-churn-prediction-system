# preprocessor.py
# Cleans and prepares the raw Telco Customer Churn dataset for machine learning.
# Handles the known data issues (TotalCharges spaces, useless customerID,
# Yes/No labels), encodes categorical variables, splits into train/test,
# and scales numerical features.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(df):
    """
    Full preprocessing pipeline for the Telco Customer Churn dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        The raw DataFrame as returned by data_loader.load_and_analyze().

    Returns
    -------
    X_train : pandas.DataFrame
        Training features (scaled).
    X_test : pandas.DataFrame
        Test features (scaled).
    y_train : pandas.Series
        Training target (0/1).
    y_test : pandas.Series
        Test target (0/1).
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler (needed later to transform new data the same way).
    feature_names : list
        List of feature column names after encoding.
    df_processed : pandas.DataFrame
        The fully cleaned + encoded DataFrame (before train/test split).
    """

    # Work on a copy so we never mutate the caller's DataFrame.
    df = df.copy()

    # ==================================================================
    # STEP 1 - Fix TotalCharges
    # ------------------------------------------------------------------
    # In the raw file, some TotalCharges cells hold a single space " "
    # instead of a real NaN. Because of that, pandas read the column as
    # text (object) and we cannot do math on it yet.
    # We:
    #   (a) replace those spaces / empty strings with NaN,
    #   (b) convert the column to float,
    #   (c) fill the NaNs with the median TotalCharges.
    # NOTE: rows with blank TotalCharges are typically brand-new customers
    # (tenure = 0) who haven't been billed a full cycle yet.
    # ==================================================================
    print("STEP 1: Fixing TotalCharges...")

    # Strip whitespace and turn empty strings into NaN.
    df["TotalCharges"] = df["TotalCharges"].astype(str).str.strip()
    df["TotalCharges"] = df["TotalCharges"].replace("", np.nan)

    # Convert to numeric; anything that still can't be parsed becomes NaN.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing values with the median (robust to outliers).
    median_total = df["TotalCharges"].median()
    missing_before = df["TotalCharges"].isnull().sum()
    df["TotalCharges"] = df["TotalCharges"].fillna(median_total)
    print(f"  Filled {missing_before} missing TotalCharges with median = {median_total:.2f}")

    # ==================================================================
    # STEP 2 - Drop useless columns
    # ------------------------------------------------------------------
    # customerID is just a unique identifier. It carries no predictive
    # signal and would only confuse the model, so we drop it.
    # ==================================================================
    print("STEP 2: Dropping customerID...")
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # ==================================================================
    # STEP 3 - Encode the target variable (Churn)
    # ------------------------------------------------------------------
    # Models need numbers, not strings. We map:
    #   "Yes" -> 1  (customer churned)
    #   "No"  -> 0  (customer stayed)
    # ==================================================================
    print("STEP 3: Encoding target variable Churn (Yes->1, No->0)...")
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)

    # ==================================================================
    # STEP 4 - Label-encode binary categorical columns
    # ------------------------------------------------------------------
    # These columns only have two possible values, so we map them
    # directly to 0 / 1. No need for one-hot encoding here.
    # ==================================================================
    print("STEP 4: Label-encoding binary columns...")
    binary_mappings = {
        "gender":           {"Female": 0, "Male": 1},
        "Partner":          {"No": 0, "Yes": 1},
        "Dependents":       {"No": 0, "Yes": 1},
        "PhoneService":     {"No": 0, "Yes": 1},
        "PaperlessBilling": {"No": 0, "Yes": 1},
    }
    for col, mapping in binary_mappings.items():
        df[col] = df[col].map(mapping).astype(int)

    # ==================================================================
    # STEP 5 - One-hot encode multi-class categorical columns
    # ------------------------------------------------------------------
    # These columns have 3+ possible values. We turn each value into its
    # own 0/1 column. drop_first=True drops one category per column to
    # avoid the "dummy variable trap" (perfect multicollinearity).
    #
    # NOTE: Contract and PaymentMethod are intentionally NOT in this
    # list. feature_engineer.py already converts them into ordinal
    # risk scores (contract_risk and payment_risk), so we drop the
    # original string columns instead of one-hot encoding them again.
    # That avoids duplicate signal and keeps the feature space smaller.
    # ==================================================================
    print("STEP 5: One-hot encoding multi-class columns...")
    multi_class_cols = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    df = pd.get_dummies(df, columns=multi_class_cols, drop_first=True)

    # Drop the original Contract / PaymentMethod string columns if they
    # are still present (they have already been encoded as
    # contract_risk / payment_risk during feature engineering).
    for col in ["Contract", "PaymentMethod"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # get_dummies produces boolean columns in newer pandas versions.
    # Cast them to int so every feature is numeric (0/1).
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # Snapshot the fully cleaned + encoded DataFrame before splitting.
    df_processed = df.copy()

    # ==================================================================
    # STEP 6 - Feature / target split
    # ------------------------------------------------------------------
    # X = all input features (everything except the label).
    # y = the label / target we want to predict.
    # ==================================================================
    print("STEP 6: Splitting into features (X) and target (y)...")
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    feature_names = X.columns.tolist()

    # ==================================================================
    # STEP 7 - Train / test split
    # ------------------------------------------------------------------
    # We hold out 20% of the data to honestly evaluate the model on
    # examples it has never seen during training.
    # stratify=y keeps the churn ratio (~26.5%) the same in train and
    # test, which matters for imbalanced classification.
    # random_state=42 makes the split reproducible.
    # ==================================================================
    print("STEP 7: Train/test split (test_size=0.2, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print(f"  X_train shape : {X_train.shape}")
    print(f"  X_test  shape : {X_test.shape}")
    print(f"  y_train shape : {y_train.shape}")
    print(f"  y_test  shape : {y_test.shape}")

    # ==================================================================
    # STEP 8 - Feature scaling
    # ------------------------------------------------------------------
    # Numerical columns live on very different scales (tenure ~ 0-72,
    # MonthlyCharges ~ 18-120, TotalCharges ~ 0-8700). Some models
    # (logistic regression, SVM, KNN) work much better when features
    # have similar scale. We standardize to mean=0, std=1.
    #
    # IMPORTANT: fit ONLY on the training set, then transform both sets.
    # Fitting on the test set would leak information from it.
    # The 0/1 columns (binary + one-hot) are left as-is.
    # ==================================================================
    print("STEP 8: Scaling numerical features (tenure, MonthlyCharges, TotalCharges)...")
    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    print("Preprocessing complete.\n")

    return X_train, X_test, y_train, y_test, scaler, feature_names, df_processed


if __name__ == "__main__":
    # Quick smoke test: load the raw data and run preprocessing on it.
    from data_loader import load_and_analyze

    raw_df = load_and_analyze("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    X_train, X_test, y_train, y_test, scaler, feature_names, df_processed = preprocess_data(raw_df)

    print("Final processed feature count :", len(feature_names))
    print("Sample feature names          :", feature_names[:10], "...")
