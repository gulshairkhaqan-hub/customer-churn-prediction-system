# feature_engineer.py
# Creates business-driven features from the raw Telco Customer Churn data.
#
# This module runs BEFORE the encoding/scaling step in preprocessor.py, so the
# input still has its original string categories ("Yes" / "No" / "Month-to-month"
# / etc.). We use those human-readable values to build features that capture
# domain knowledge about why customers churn.

import numpy as np
import pandas as pd


def engineer_features(df):
    """
    Add business-driven features to the raw Telco Customer Churn DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw DataFrame (as loaded by data_loader.load_and_analyze).
        Categorical columns should still hold their original strings
        (e.g., "Yes" / "No" / "Month-to-month").

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with 7 additional engineered feature columns.
    """

    # Work on a copy so we never mutate the caller's DataFrame.
    df = df.copy()

    # ------------------------------------------------------------------
    # Safety step: force TotalCharges to a real numpy float64 column.
    # ------------------------------------------------------------------
    # Two problems we have to defuse here BEFORE any feature is built:
    #   1. The raw file stores some TotalCharges cells as a single space
    #      " ", so the column is read as text instead of a number.
    #   2. If pandas is using the pyarrow backend, the column may have
    #      dtype 'large_string', and dividing it by an int column raises:
    #        TypeError: operation 'truediv' not supported for dtype 'str'
    #                   with dtype 'int64'
    #
    # Fix:
    #   - cast to str, strip whitespace, turn "" into NaN
    #   - pd.to_numeric(..., errors="coerce") converts to a numeric type
    #   - .astype("float64") forces a plain numpy float64 dtype, which
    #     bypasses any arrow-backed dtype and makes division safe.
    # We do NOT permanently fill NaNs here — that's the preprocessor's job.
    # ------------------------------------------------------------------
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip().replace("", np.nan),
        errors="coerce",
    ).astype("float64")

    # ==================================================================
    # FEATURE 1 - tenure_group (Customer Lifecycle Stage)
    # ------------------------------------------------------------------
    # Business logic: brand-new customers haven't built loyalty yet and
    # churn at much higher rates. Long-tenured customers are "sticky".
    # We bucket tenure (in months) into four lifecycle stages and encode:
    #   New          -> 0
    #   Early Stage  -> 1
    #   Mid Stage    -> 2
    #   Loyal        -> 3
    # ==================================================================
    def _tenure_to_group(months):
        if months <= 12:
            return 0   # "New Customer"
        elif months <= 24:
            return 1   # "Early Stage"
        elif months <= 48:
            return 2   # "Mid Stage"
        else:
            return 3   # "Loyal Customer"

    df["tenure_group"] = df["tenure"].apply(_tenure_to_group).astype(int)

    # ==================================================================
    # FEATURE 2 - charges_per_month_ratio
    # ------------------------------------------------------------------
    # Business logic: TotalCharges divided by (tenure + 1) gives an
    # "effective monthly spend" smoothed over the customer's lifetime.
    # If this jumps far above the customer's typical MonthlyCharges, it
    # suggests recent billing spikes — a classic churn warning sign.
    # We add +1 to tenure to avoid division-by-zero for brand-new customers.
    # ==================================================================
    df["charges_per_month_ratio"] = df["TotalCharges"] / (df["tenure"] + 1)

    # ==================================================================
    # FEATURE 3 - has_multiple_services (Service Bundle Score)
    # ------------------------------------------------------------------
    # Business logic: the more services a customer subscribes to, the
    # higher the switching cost — they are less likely to churn.
    # We count how many of these six add-on services are "Yes".
    # Result is an integer from 0 (no add-ons) to 6 (all add-ons).
    # ==================================================================
    service_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    df["has_multiple_services"] = (
        df[service_cols].apply(lambda col: col == "Yes").sum(axis=1).astype(int)
    )

    # ==================================================================
    # FEATURE 4 - is_high_value
    # ------------------------------------------------------------------
    # Business logic: high-paying customers are valuable but also more
    # demanding. If they don't feel they're getting their money's worth,
    # they leave. We flag MonthlyCharges > 70 as "high value".
    # ==================================================================
    df["is_high_value"] = (df["MonthlyCharges"] > 70).astype(int)

    # ==================================================================
    # FEATURE 5 - contract_risk
    # ------------------------------------------------------------------
    # Business logic: contract length is one of the strongest churn
    # predictors. Month-to-month customers can walk away anytime,
    # while two-year contract customers are locked in.
    #   Month-to-month -> 3  (highest risk)
    #   One year       -> 2
    #   Two year       -> 1  (lowest risk)
    # ==================================================================
    contract_risk_map = {
        "Month-to-month": 3,
        "One year": 2,
        "Two year": 1,
    }
    df["contract_risk"] = df["Contract"].map(contract_risk_map).astype(int)

    # ==================================================================
    # FEATURE 6 - payment_risk
    # ------------------------------------------------------------------
    # Business logic: in this dataset, electronic check users churn far
    # more than auto-pay users — likely because they re-evaluate their
    # bill every cycle (more billing friction = more chances to leave).
    #   Electronic check            -> 2 (highest risk)
    #   Mailed check                -> 1
    #   Bank transfer / Credit card -> 0 (lowest risk, auto-pay)
    # ==================================================================
    payment_risk_map = {
        "Electronic check":            2,
        "Mailed check":                1,
        "Bank transfer (automatic)":   0,
        "Credit card (automatic)":     0,
    }
    df["payment_risk"] = df["PaymentMethod"].map(payment_risk_map).astype(int)

    # ==================================================================
    # FEATURE 7 - no_support_services
    # ------------------------------------------------------------------
    # Business logic: customers without TechSupport AND without
    # OnlineSecurity feel unsupported when something goes wrong. They
    # are far more likely to churn after a single bad experience.
    # Flag = 1 if BOTH services are "No".
    # ==================================================================
    df["no_support_services"] = (
        (df["TechSupport"] == "No") & (df["OnlineSecurity"] == "No")
    ).astype(int)

    # ------------------------------------------------------------------
    # Print a quick summary of the new features so we can sanity-check.
    # ------------------------------------------------------------------
    new_features = [
        "tenure_group",
        "charges_per_month_ratio",
        "has_multiple_services",
        "is_high_value",
        "contract_risk",
        "payment_risk",
        "no_support_services",
    ]

    print("=" * 70)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 70)
    print(f"Created {len(new_features)} new features:\n")
    for feat in new_features:
        col = df[feat]
        if pd.api.types.is_float_dtype(col):
            print(
                f"  - {feat:<25} | dtype={str(col.dtype):<8} "
                f"| min={col.min():.2f}  max={col.max():.2f}  mean={col.mean():.2f}"
            )
        else:
            unique_vals = sorted(col.unique().tolist())
            print(
                f"  - {feat:<25} | dtype={str(col.dtype):<8} "
                f"| min={col.min()}  max={col.max()}  unique={unique_vals}"
            )
    print()

    return df


if __name__ == "__main__":
    # Quick smoke test: load the raw data and run feature engineering on it.
    from data_loader import load_and_analyze

    raw_df = load_and_analyze("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df_with_features = engineer_features(raw_df)

    print("Final column count:", df_with_features.shape[1])
    print("New features preview:")
    print(
        df_with_features[
            [
                "tenure",
                "tenure_group",
                "charges_per_month_ratio",
                "has_multiple_services",
                "is_high_value",
                "contract_risk",
                "payment_risk",
                "no_support_services",
            ]
        ].head(10)
    )
