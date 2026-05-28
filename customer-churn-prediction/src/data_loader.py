# data_loader.py
# Loads the raw Telco Customer Churn dataset from the data/ directory and
# performs a first-pass exploratory analysis (shape, dtypes, missing values,
# duplicates, churn distribution, basic stats) to help us understand the data
# before any preprocessing or modeling.

import pandas as pd


def load_and_analyze(filepath):
    """
    Load the Telco Customer Churn dataset and print a quick exploratory summary.

    Parameters
    ----------
    filepath : str
        Path to the CSV file (e.g., "data/WA_Fn-UseC_-Telco-Customer-Churn.csv").

    Returns
    -------
    pandas.DataFrame
        The loaded raw DataFrame (no cleaning applied here).
    """

    # ------------------------------------------------------------------
    # 1) Load the CSV file into a pandas DataFrame.
    #    pandas reads the file into a 2D table (rows = customers, cols = features).
    # ------------------------------------------------------------------
    print("=" * 70)
    print("LOADING DATASET")
    print("=" * 70)
    print(f"Reading file: {filepath}")

    df = pd.read_csv(filepath)
    print("File loaded successfully.\n")

    # ------------------------------------------------------------------
    # 2) Print the total number of rows and columns.
    #    df.shape returns a tuple: (rows, columns).
    # ------------------------------------------------------------------
    print("=" * 70)
    print("DATASET SHAPE")
    print("=" * 70)
    rows, cols = df.shape
    print(f"Total rows    : {rows}")
    print(f"Total columns : {cols}\n")

    # ------------------------------------------------------------------
    # 3) Print all column names and their data types.
    #    df.dtypes shows what kind of values each column holds
    #    (e.g., int64, float64, object/string).
    # ------------------------------------------------------------------
    print("=" * 70)
    print("COLUMN NAMES AND DATA TYPES")
    print("=" * 70)
    print(df.dtypes)
    print()

    # ------------------------------------------------------------------
    # 4) Print the first 5 rows so we can eyeball what the data looks like.
    # ------------------------------------------------------------------
    print("=" * 70)
    print("FIRST 5 ROWS")
    print("=" * 70)
    print(df.head())
    print()

    # ------------------------------------------------------------------
    # 5) Print missing values per column.
    #    isnull() marks empty cells as True; sum() counts them per column.
    # ------------------------------------------------------------------
    print("=" * 70)
    print("MISSING VALUES PER COLUMN")
    print("=" * 70)
    missing = df.isnull().sum()
    print(missing)
    print(f"\nTotal missing values across all columns: {missing.sum()}\n")

    # ------------------------------------------------------------------
    # 6) Print the number of fully duplicated rows.
    #    duplicated() flags rows that are exact copies of an earlier row.
    # ------------------------------------------------------------------
    print("=" * 70)
    print("DUPLICATE ROWS")
    print("=" * 70)
    duplicate_count = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicate_count}\n")

    # ------------------------------------------------------------------
    # 7) Detect the "hidden missing values" in TotalCharges.
    #    In this dataset, some TotalCharges cells contain a single space (" ")
    #    instead of a real NaN. Because of that, pandas reads the whole column
    #    as text (object) and isnull() does NOT count them as missing.
    #    We catch them by checking for stripped-empty strings.
    # ------------------------------------------------------------------
    print("=" * 70)
    print("HIDDEN MISSING VALUES IN 'TotalCharges'")
    print("=" * 70)
    if "TotalCharges" in df.columns:
        # Convert to string, strip whitespace, then check for empty strings.
        blank_mask = df["TotalCharges"].astype(str).str.strip() == ""
        blank_count = blank_mask.sum()
        print(f"TotalCharges dtype                       : {df['TotalCharges'].dtype}")
        print(f"Rows where TotalCharges is blank/space   : {blank_count}")
        print("(These look 'present' to pandas but are actually missing values.)\n")
    else:
        print("Column 'TotalCharges' not found in the dataset.\n")

    # ------------------------------------------------------------------
    # 8) Print churn distribution.
    #    The target column 'Churn' has values 'Yes' (left) and 'No' (stayed).
    # ------------------------------------------------------------------
    print("=" * 70)
    print("CHURN DISTRIBUTION")
    print("=" * 70)
    if "Churn" in df.columns:
        churn_counts = df["Churn"].value_counts()
        churned = int(churn_counts.get("Yes", 0))
        stayed = int(churn_counts.get("No", 0))
        total = churned + stayed
        churn_rate = (churned / total) * 100 if total > 0 else 0.0

        print(f"Churned (Yes) : {churned}")
        print(f"Stayed  (No)  : {stayed}")
        print(f"Churn rate    : {churn_rate:.2f}%\n")
    else:
        print("Column 'Churn' not found in the dataset.\n")

    # ------------------------------------------------------------------
    # 9) Identify numerical vs categorical columns.
    #    - Numerical columns hold numbers (int / float).
    #    - Categorical columns hold labels/text (object).
    # ------------------------------------------------------------------
    print("=" * 70)
    print("NUMERICAL vs CATEGORICAL COLUMNS")
    print("=" * 70)
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    print(f"Numerical columns ({len(numerical_cols)}):")
    for c in numerical_cols:
        print(f"  - {c}")
    print(f"\nCategorical columns ({len(categorical_cols)}):")
    for c in categorical_cols:
        print(f"  - {c}")
    print()

    # ------------------------------------------------------------------
    # 10) Basic statistics for tenure, MonthlyCharges, TotalCharges.
    #     describe() reports count, mean, std, min, quartiles, and max.
    #     TotalCharges is currently text because of the blank values, so we
    #     convert it to numeric just for the stats view (errors='coerce' turns
    #     any non-numeric value into NaN). The original df is untouched.
    # ------------------------------------------------------------------
    print("=" * 70)
    print("BASIC STATISTICS (tenure, MonthlyCharges, TotalCharges)")
    print("=" * 70)
    stats_df = df[["tenure", "MonthlyCharges", "TotalCharges"]].copy()
    stats_df["TotalCharges"] = pd.to_numeric(stats_df["TotalCharges"], errors="coerce")
    print(stats_df.describe())
    print()

    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    # Return the raw DataFrame so callers (e.g., main.py) can keep using it.
    return df


if __name__ == "__main__":
    df = load_and_analyze("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
