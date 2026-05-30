# visualizer.py
# Visualization module for the Customer Churn Prediction system.
#
# Every function in this module saves a PNG to the outputs/ folder and
# prints a confirmation line. We use a dark grid style across the board
# so the plots feel consistent in a dashboard or report.

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


# Folder where all charts are saved.
OUTPUTS_DIR = "outputs"

# Apply the requested style globally for every plot in this module.
plt.style.use("seaborn-v0_8-darkgrid")


def _ensure_outputs_dir():
    """Create the outputs/ folder if it doesn't exist yet."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)


def _save_and_close(fig, path):
    """Save the figure, print a confirmation, and close the canvas."""
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


# ======================================================================
# PLOT 1: Churn distribution (countplot)
# ======================================================================
def plot_churn_distribution(df):
    """
    Show how many customers churned vs stayed.
    This makes the class imbalance visible at a glance — only ~26.5%
    of customers churn, which matters a lot for model evaluation.
    """
    _ensure_outputs_dir()
    save_path = os.path.join(OUTPUTS_DIR, "churn_distribution.png")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(data=df, x="Churn", ax=ax, palette=["#2ecc71", "#e74c3c"])

    # Add percentage labels on top of each bar.
    total = len(df)
    for patch in ax.patches:
        height = patch.get_height()
        pct = (height / total) * 100
        ax.annotate(
            f"{int(height)}\n({pct:.1f}%)",
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_title("Customer Churn Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Number of customers")

    _save_and_close(fig, save_path)
    return save_path


# ======================================================================
# PLOT 2: Correlation heatmap (numerical columns only)
# ======================================================================
def plot_correlation_heatmap(df):
    """
    Heatmap of pairwise correlations between numerical columns.
    Helps spot redundant features and strong relationships with churn.
    """
    _ensure_outputs_dir()
    save_path = os.path.join(OUTPUTS_DIR, "correlation_heatmap.png")

    # Make a numeric-only copy. TotalCharges may still be text in the raw
    # DataFrame, so we coerce defensively without mutating the caller's df.
    numeric_df = df.copy()
    if "TotalCharges" in numeric_df.columns and numeric_df["TotalCharges"].dtype == object:
        numeric_df["TotalCharges"] = pd.to_numeric(
            numeric_df["TotalCharges"].astype(str).str.strip().replace("", np.nan),
            errors="coerce",
        )

    # Encode Churn to 0/1 for correlation purposes (only if it's still text).
    if "Churn" in numeric_df.columns and numeric_df["Churn"].dtype == object:
        numeric_df["Churn"] = numeric_df["Churn"].map({"No": 0, "Yes": 1})

    numeric_df = numeric_df.select_dtypes(include=[np.number])

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        numeric_df.corr(),
        annot=True, fmt=".2f", cmap="RdYlGn",
        ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")

    _save_and_close(fig, save_path)
    return save_path


# ======================================================================
# PLOT 3: Churn by contract type
# ======================================================================
def plot_churn_by_contract(df):
    """
    Side-by-side counts of churned vs retained customers, broken down
    by contract type. Month-to-month customers churn the most because
    they can leave anytime.
    """
    _ensure_outputs_dir()
    save_path = os.path.join(OUTPUTS_DIR, "churn_by_contract.png")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(
        data=df, x="Contract", hue="Churn",
        palette=["#2ecc71", "#e74c3c"], ax=ax,
    )
    ax.set_title("Churn Rate by Contract Type", fontsize=14, fontweight="bold")
    ax.set_xlabel("Contract type")
    ax.set_ylabel("Number of customers")
    ax.legend(title="Churn")

    _save_and_close(fig, save_path)
    return save_path


# ======================================================================
# PLOT 4: Tenure distribution split by churn
# ======================================================================
def plot_churn_by_tenure(df):
    """
    Histogram of customer tenure (in months), split by churn outcome.
    Reveals that churn is concentrated in low-tenure customers.
    """
    _ensure_outputs_dir()
    save_path = os.path.join(OUTPUTS_DIR, "churn_by_tenure.png")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(
        data=df, x="tenure", hue="Churn", multiple="stack",
        bins=30, palette=["#2ecc71", "#e74c3c"], ax=ax,
    )
    ax.set_title("Tenure Distribution: Churned vs Retained", fontsize=14, fontweight="bold")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Number of customers")

    _save_and_close(fig, save_path)
    return save_path


# ======================================================================
# PLOT 5: Monthly charges vs churn (boxplot)
# ======================================================================
def plot_monthly_charges_vs_churn(df):
    """
    Boxplot of MonthlyCharges grouped by churn outcome. Churners
    typically pay higher monthly bills than customers who stay.
    """
    _ensure_outputs_dir()
    save_path = os.path.join(OUTPUTS_DIR, "charges_vs_churn.png")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=df, x="Churn", y="MonthlyCharges",
        palette=["#2ecc71", "#e74c3c"], ax=ax,
    )
    ax.set_title("Monthly Charges: Churned vs Retained Customers",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Monthly charges ($)")

    _save_and_close(fig, save_path)
    return save_path


# ======================================================================
# PLOT 6: Model comparison bar chart
# ======================================================================
def plot_model_comparison(results_dict):
    """
    Grouped bar chart comparing AUC-ROC, F1, Precision, Recall across
    all three trained models.

    Parameters
    ----------
    results_dict : dict
        The all_results dict returned by model_trainer.train_and_compare().
        Each value must contain a "metrics" sub-dict.
    """
    _ensure_outputs_dir()
    save_path = os.path.join(OUTPUTS_DIR, "model_comparison.png")

    metrics_to_plot = ["AUC-ROC", "F1", "Precision", "Recall"]
    model_names = list(results_dict.keys())

    # Build a 2D matrix: rows = metrics, columns = models.
    values = np.array([
        [results_dict[m]["metrics"][k] for m in model_names]
        for k in metrics_to_plot
    ])

    x = np.arange(len(model_names))
    width = 0.2
    colors = ["#3498db", "#9b59b6", "#f39c12", "#1abc9c"]

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, metric in enumerate(metrics_to_plot):
        offset = (i - (len(metrics_to_plot) - 1) / 2) * width
        bars = ax.bar(x + offset, values[i], width, label=metric, color=colors[i])
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.3f}",
                (bar.get_x() + bar.get_width() / 2, h),
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")

    _save_and_close(fig, save_path)
    return save_path


# ======================================================================
# PLOT 7: Confusion matrix heatmap for a single model
# ======================================================================
def plot_confusion_matrix(y_test, y_pred, model_name):
    """
    Confusion matrix as a heatmap with labeled axes.
    The filename is slugified so model names with spaces or special
    characters still produce a valid path.
    """
    _ensure_outputs_dir()

    # Slug-safe filename: lowercase, spaces -> underscores.
    slug = (
        str(model_name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
    )
    save_path = os.path.join(OUTPUTS_DIR, f"confusion_matrix_{slug}.png")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Stayed", "Churned"],
        yticklabels=["Stayed", "Churned"],
        ax=ax, cbar=True,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")

    _save_and_close(fig, save_path)
    return save_path


# ======================================================================
# PLOT 8: Risk segmentation pie chart
# ======================================================================
def plot_risk_segmentation(scoring_df):
    """
    Pie chart of HIGH / MEDIUM / LOW risk customer counts from the
    weekly scoring DataFrame.
    """
    _ensure_outputs_dir()
    save_path = os.path.join(OUTPUTS_DIR, "risk_segmentation.png")

    # Force a consistent order so colors always map the same way.
    order = ["HIGH", "MEDIUM", "LOW"]
    counts = (
        scoring_df["risk_level"]
        .value_counts()
        .reindex(order)
        .fillna(0)
        .astype(int)
    )
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(12, 6))
    # Hide segments that are exactly zero so the pie doesn't look broken.
    nonzero = counts[counts > 0]
    nonzero_colors = [colors[order.index(label)] for label in nonzero.index]

    ax.pie(
        nonzero.values,
        labels=nonzero.index,
        colors=nonzero_colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 12, "fontweight": "bold"},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    ax.set_title("Customer Risk Segmentation", fontsize=14, fontweight="bold")
    ax.axis("equal")

    _save_and_close(fig, save_path)
    return save_path


# ======================================================================
# PLOT 9: Churn probability distribution with threshold lines
# ======================================================================
def plot_churn_probability_distribution(scoring_df):
    """
    Histogram of predicted churn probabilities with vertical lines at
    the 0.40 (MEDIUM) and 0.70 (HIGH) risk thresholds, so it's obvious
    where the segments split.
    """
    _ensure_outputs_dir()
    save_path = os.path.join(OUTPUTS_DIR, "churn_prob_distribution.png")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(
        scoring_df["churn_probability"], bins=30, kde=True,
        color="#3498db", ax=ax,
    )
    ax.axvline(0.40, color="#f39c12", linestyle="--", linewidth=2,
               label="MEDIUM threshold (0.40)")
    ax.axvline(0.70, color="#e74c3c", linestyle="--", linewidth=2,
               label="HIGH threshold (0.70)")
    ax.set_title("Churn Probability Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Churn probability")
    ax.set_ylabel("Number of customers")
    ax.legend()

    _save_and_close(fig, save_path)
    return save_path


if __name__ == "__main__":
    # Smoke test: run the full pipeline and produce every plot.
    from data_loader import load_and_analyze
    from feature_engineer import engineer_features
    from preprocessor import preprocess_data
    from model_trainer import train_and_compare
    from scorer import run_weekly_scoring

    raw_df = load_and_analyze("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # EDA plots use the raw DataFrame.
    plot_churn_distribution(raw_df)
    plot_correlation_heatmap(raw_df)
    plot_churn_by_contract(raw_df)
    plot_churn_by_tenure(raw_df)
    plot_monthly_charges_vs_churn(raw_df)

    # Modeling plots need a trained model + results.
    df_feat = engineer_features(raw_df)
    X_train, X_test, y_train, y_test, scaler, feature_names, df_processed = preprocess_data(df_feat)

    best_model, all_results, model_name = train_and_compare(
        X_train, X_test, y_train, y_test, feature_names
    )

    plot_model_comparison(all_results)
    for name, result in all_results.items():
        plot_confusion_matrix(y_test, result["y_pred"], name)

    # Scoring-stage plots need the scoring_df.
    scoring_df = run_weekly_scoring(best_model, scaler, X_test, feature_names, week_number=1)
    plot_risk_segmentation(scoring_df)
    plot_churn_probability_distribution(scoring_df)
