# Requirements Document

## Introduction

The Customer Churn Prediction System is an end-to-end machine learning application that predicts which telecom customers are most likely to cancel their service, explains why each customer is at risk, segments customers into actionable risk tiers, and recommends concrete retention actions. It is built on the IBM Telco Customer Churn dataset (7,043 customers, 20 features) and consists of an offline training/scoring pipeline (`main.py` orchestrating modules under `src/`) and an interactive read-only Streamlit dashboard (`app.py`).

This document captures the requirements of the **existing, implemented system** ("as-is") as testable acceptance criteria, and then defines a **roadmap** of future-improvement requirements that are not yet implemented. Roadmap requirements (Requirements 12–16) are clearly marked as not-yet-implemented so they can be prioritized separately from the documented current behavior.

The primary goal is to give a retention team a ranked list of at-risk customers and a specific next action for each, rather than treating every customer the same.

## Glossary

- **Churn**: A customer canceling their service (target column `Churn`, Yes/No).
- **Churn probability**: Model-predicted probability (0.0–1.0) that a customer will churn.
- **Risk tier**: HIGH (≥ 0.70), MEDIUM (0.40–0.69), LOW (< 0.40).
- **Pipeline**: The offline run (`main.py`) that loads data, engineers features, preprocesses, trains, explains, scores, recommends, and saves artifacts.
- **Artifact**: A saved output — a model `.pkl` in `models/`, or a CSV/PNG in `outputs/`.
- **SHAP**: SHapley Additive exPlanations, used to attribute each prediction to its driving features.
- **ROADMAP**: A label used in some requirement titles to indicate a planned, not-yet-implemented capability.

---

## Requirements

### Requirement 1: Data Loading and Validation

**User Story:** As a data scientist, I want the raw Telco dataset loaded and validated, so that I can trust the data before training models on it.

#### Acceptance Criteria

1. WHEN the pipeline runs THEN the system SHALL load the dataset from `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` into a pandas DataFrame.
2. IF the dataset file does not exist at the configured path THEN the system SHALL report a clear error identifying the missing file and SHALL stop the pipeline.
3. WHEN the dataset is loaded THEN the system SHALL report row count, column count, column names, and data types.
4. WHEN the dataset is loaded THEN the system SHALL report the count of missing values per column and the count of fully duplicated rows.
5. WHEN analyzing `TotalCharges` THEN the system SHALL detect values that are blank or whitespace-only (which pandas does not treat as NaN) and SHALL report how many rows are affected.
6. WHEN analyzing the target THEN the system SHALL report the count of churned (`Yes`) and retained (`No`) customers and the churn rate as a percentage.

### Requirement 2: Feature Engineering

**User Story:** As a data scientist, I want business-driven features derived from the raw data, so that the model can capture domain knowledge about churn drivers.

#### Acceptance Criteria

1. WHEN feature engineering runs THEN the system SHALL operate on the raw (pre-encoding) data so that original categorical string values are available.
2. WHEN feature engineering runs THEN the system SHALL coerce `TotalCharges` to a numeric `float64` type before any feature uses it for arithmetic.
3. WHEN feature engineering runs THEN the system SHALL create `tenure_group` as an ordinal lifecycle stage encoded 0–3 (0=New ≤12 months, 1=Early 13–24, 2=Mid 25–48, 3=Loyal 49+).
4. WHEN feature engineering runs THEN the system SHALL create `charges_per_month_ratio` as `TotalCharges / (tenure + 1)`.
5. WHEN feature engineering runs THEN the system SHALL create `has_multiple_services` as the integer count (0–6) of add-on services held among OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, and StreamingMovies.
6. WHEN feature engineering runs THEN the system SHALL create `is_high_value` as 1 if `MonthlyCharges > 70` else 0.
7. WHEN feature engineering runs THEN the system SHALL create `contract_risk` as an ordinal score (Month-to-month=3, One year=2, Two year=1).
8. WHEN feature engineering runs THEN the system SHALL create `payment_risk` as an ordinal score (Electronic check=2, Mailed check=1, Bank transfer/Credit card=0).
9. WHEN feature engineering runs THEN the system SHALL create `no_support_services` as 1 if both `TechSupport` and `OnlineSecurity` are "No" else 0.
10. WHEN feature engineering completes THEN the system SHALL print a summary listing each created feature and its value range, and SHALL return the DataFrame with all engineered features added.

### Requirement 3: Data Preprocessing

**User Story:** As a data scientist, I want the data cleaned, encoded, split, and scaled consistently, so that models receive numeric, leakage-free inputs.

#### Acceptance Criteria

1. WHEN preprocessing runs THEN the system SHALL replace blank/whitespace `TotalCharges` values with NaN, convert the column to float, and fill remaining NaN with the column median.
2. WHEN preprocessing runs THEN the system SHALL drop the `customerID` column.
3. WHEN preprocessing runs THEN the system SHALL encode the target `Churn` as 1 for "Yes" and 0 for "No".
4. WHEN preprocessing runs THEN the system SHALL label-encode binary categorical columns (gender, Partner, Dependents, PhoneService, PaperlessBilling) to 0/1.
5. WHEN preprocessing runs THEN the system SHALL one-hot encode multi-class categorical columns with `drop_first=True`, EXCLUDING `Contract` and `PaymentMethod` because they are already represented as `contract_risk` and `payment_risk`.
6. IF the original `Contract` or `PaymentMethod` string columns are still present THEN the system SHALL drop them after encoding.
7. WHEN preprocessing runs THEN the system SHALL split features and target into train/test sets with `test_size=0.2`, `random_state=42`, and stratification on the target.
8. WHEN preprocessing runs THEN the system SHALL fit a `StandardScaler` on the numerical columns (tenure, MonthlyCharges, TotalCharges) using the training set only, and SHALL apply that fitted scaler to both train and test sets.
9. WHEN preprocessing completes THEN the system SHALL return X_train, X_test, y_train, y_test, the fitted scaler, the feature names, and the fully processed DataFrame.

### Requirement 4: Model Training and Selection

**User Story:** As a data scientist, I want multiple models trained, compared, and the best selected, so that the system uses the most effective model for ranking churn risk.

#### Acceptance Criteria

1. WHEN training runs THEN the system SHALL train three models: Logistic Regression (baseline), XGBoost, and LightGBM.
2. WHEN training runs THEN the system SHALL handle class imbalance for each model (Logistic Regression and LightGBM via balanced class weights, XGBoost via `scale_pos_weight = negatives/positives`).
3. IF any input feature value is NaN at training time THEN the system SHALL replace it with 0 before fitting, while preserving feature/column names.
4. WHEN evaluating each model THEN the system SHALL compute AUC-ROC, Precision, Recall, F1, and Accuracy on the test set.
5. WHEN evaluation completes THEN the system SHALL print a comparison table of all metrics for all models and a confusion matrix for each model.
6. WHEN selecting the best model THEN the system SHALL choose the model with the highest test-set AUC-ROC and SHALL report the selected model name and its AUC-ROC.
7. WHEN training completes THEN the system SHALL save each model to `models/` (`logistic_regression.pkl`, `xgboost_model.pkl`, `lightgbm_model.pkl`) and SHALL save the selected model as `models/best_model.pkl`.

### Requirement 5: Model Explainability (SHAP)

**User Story:** As a retention analyst, I want to see why the model flags customers as at-risk, so that I can trust and act on the predictions.

#### Acceptance Criteria

1. WHEN explanations are generated THEN the system SHALL select a SHAP explainer appropriate to the model type (TreeExplainer for XGBoost/LightGBM, LinearExplainer for Logistic Regression).
2. IF the primary SHAP explainer fails THEN the system SHALL fall back to a general-purpose SHAP explainer rather than crashing.
3. WHEN SHAP values are computed THEN the system SHALL normalize them to a 2D array for the positive (churn) class regardless of SHAP version or output shape.
4. WHEN a global explanation is requested THEN the system SHALL produce a summary plot of the top 15 churn drivers and SHALL save it to `outputs/shap_summary.png`.
5. WHEN a single customer is explained THEN the system SHALL return the top 3 most influential features, their SHAP values, and whether each increases or decreases churn risk.
6. WHEN a human-readable explanation is requested THEN the system SHALL convert the structured explanation into plain-language text using business-friendly feature names.

### Requirement 6: Weekly Batch Scoring and Risk Segmentation

**User Story:** As a retention manager, I want all customers scored and segmented into risk tiers each week, so that my team can prioritize outreach.

#### Acceptance Criteria

1. WHEN scoring runs THEN the system SHALL compute a churn probability for every customer using the selected model's positive-class probability.
2. IF any input value is NaN at scoring time THEN the system SHALL replace it with 0 before predicting.
3. WHEN assigning risk tiers THEN the system SHALL classify probability ≥ 0.70 as HIGH, 0.40–0.69 as MEDIUM, and < 0.40 as LOW.
4. WHEN scoring runs THEN the system SHALL produce a scoring table with columns customer_index, churn_probability (rounded to 4 decimals), risk_level, scoring_week, and scoring_date (current date).
5. WHEN scoring runs THEN the system SHALL print the total scored count and the count and percentage of customers in each risk tier.
6. WHEN scoring completes THEN the system SHALL save the scoring table to `outputs/weekly_scoring_week_{week_number}.csv` and SHALL return it.

### Requirement 7: Retention Recommendations

**User Story:** As a retention agent, I want a concrete recommended action for each high-risk customer, so that I know what to offer them.

#### Acceptance Criteria

1. WHEN recommendations are generated THEN the system SHALL process only HIGH-risk customers.
2. WHEN identifying a customer's drivers THEN the system SHALL select the top 3 features that pushed their churn risk up, based on SHAP values.
3. WHEN mapping a recommendation THEN the system SHALL apply rule-based logic on the top driver: contract → "Offer annual contract with 20% discount"; tenure → "Assign dedicated customer success manager"; charges → "Offer discounted plan or loyalty reward"; missing support/security → "Offer free tech support upgrade for 3 months"; fiber/internet → "Investigate service quality complaints"; otherwise → "Schedule retention call within 48 hours".
4. WHEN recommendations are generated THEN the system SHALL produce a table with customer_index, risk_level, churn_probability, top_reason_1, top_reason_2, top_reason_3, and recommended_action.
5. IF there are no HIGH-risk customers THEN the system SHALL still write a CSV with the correct headers and SHALL return an empty result without error.
6. WHEN recommendations complete THEN the system SHALL save them to `outputs/retention_recommendations.csv` and SHALL print a sample of the first rows.

### Requirement 8: Visualizations

**User Story:** As a stakeholder, I want clear visualizations of the data and model results, so that I can understand churn patterns at a glance.

#### Acceptance Criteria

1. WHEN visualizations are generated THEN the system SHALL save each chart as a PNG to the `outputs/` directory.
2. WHEN EDA visualizations are generated THEN the system SHALL produce churn distribution, correlation heatmap, churn-by-contract, churn-by-tenure, and monthly-charges-vs-churn charts using the original (unencoded) categorical values.
3. WHEN model visualizations are generated THEN the system SHALL produce a model comparison chart and a confusion matrix chart per model.
4. WHEN scoring visualizations are generated THEN the system SHALL produce a risk segmentation chart and a churn probability distribution chart with threshold markers at 0.40 and 0.70.
5. WHEN saving a confusion matrix THEN the system SHALL generate a filesystem-safe filename derived from the model name.

### Requirement 9: Pipeline Orchestration

**User Story:** As an operator, I want a single command to run the whole workflow end to end, so that I can regenerate all artifacts reliably.

#### Acceptance Criteria

1. WHEN `main.py` runs THEN the system SHALL execute, in order: data loading, feature engineering, preprocessing, EDA visualizations, model training, model comparison visualizations, SHAP explainability, weekly scoring, risk visualizations, retention recommendations, and artifact saving.
2. WHEN each step runs THEN the system SHALL print a clearly labeled section header for that step.
3. IF any step raises an error THEN the system SHALL report which step failed with a clear message and SHALL stop the pipeline with a non-zero exit status.
4. WHEN resolving file paths THEN the system SHALL anchor all data, model, and output paths to the script's own directory so the pipeline runs correctly regardless of the current working directory.
5. WHEN the pipeline completes successfully THEN the system SHALL save `best_model.pkl`, `scaler.pkl`, `results_dict.pkl`, and `feature_names.pkl` to `models/` and SHALL print a completion message directing the user to launch the dashboard.

### Requirement 10: Interactive Dashboard

**User Story:** As a retention team member, I want an interactive dashboard over the saved results, so that I can explore risk and recommendations without rerunning the pipeline.

#### Acceptance Criteria

1. WHEN the dashboard loads THEN the system SHALL read only pre-saved artifacts from `outputs/` and `models/` and SHALL NOT retrain any model.
2. IF a required artifact is missing THEN the affected section SHALL degrade gracefully with an informative warning or error instead of crashing.
3. WHEN the dashboard renders THEN the system SHALL provide navigation across six views: Overview, Model Performance, Risk Segmentation, Feature Importance & SHAP, Retention Recommendations, and Customer Drill-Down.
4. WHEN the Overview view renders THEN the system SHALL display summary metrics (total scored, HIGH-risk count, predicted churn rate, average churn probability) and charts for churn distribution and risk segmentation.
5. WHEN the Risk Segmentation view renders THEN the system SHALL allow filtering by risk tier and sorting by churn probability, and SHALL show per-tier counts.
6. WHEN the Customer Drill-Down view renders THEN the system SHALL let the user select a customer index and SHALL show that customer's features, churn probability, risk tier, top reasons, and recommended action.
7. WHEN heavy data or model loads occur THEN the system SHALL cache them so navigation between views is responsive.
8. WHEN charts and tables render THEN the system SHALL use styling that keeps all text readable on the dark theme and SHALL render the navigation controls fully visible (not clipped).

### Requirement 11: Environment, Configuration, and Reproducibility

**User Story:** As a developer, I want a documented, reproducible setup, so that I can install and run the project consistently.

#### Acceptance Criteria

1. WHEN setting up the project THEN the system SHALL declare all Python dependencies in `requirements.txt`.
2. WHEN random operations occur (train/test split, model training) THEN the system SHALL fix random seeds so results are reproducible across runs.
3. WHEN the dashboard theme is applied THEN the system SHALL provide a Streamlit theme configuration consistent with the dark dashboard styling.
4. WHEN documentation is consulted THEN the `README.md` SHALL describe the dataset, features, models, risk logic, project structure, installation, and run commands.

---

## Roadmap Requirements (Not Yet Implemented)

These requirements define planned enhancements from the project roadmap. They are intentionally separated from the as-is requirements above and are out of scope for the current implementation.

### Requirement 12: Real-Time Scoring API (ROADMAP)

**User Story:** As an integration engineer, I want a real-time scoring endpoint, so that other systems can request a churn score for a customer on demand.

#### Acceptance Criteria

1. WHEN a client sends customer feature data to the scoring API THEN the system SHALL return a churn probability and risk tier in the response.
2. IF the request payload is missing required fields or has invalid types THEN the system SHALL return a validation error with a descriptive message and an appropriate HTTP status code.
3. WHEN the API starts THEN the system SHALL load the saved best model and scaler once and reuse them across requests.
4. WHEN the API is queried for health THEN the system SHALL expose a health-check endpoint reporting service and model-load status.

### Requirement 13: Production Data Storage (ROADMAP)

**User Story:** As a data engineer, I want customer and scoring data stored in a relational database, so that results are durable and queryable beyond flat files.

#### Acceptance Criteria

1. WHEN scoring results are produced THEN the system SHALL persist them to a PostgreSQL database in addition to or instead of CSV files.
2. WHEN reading data for scoring or the dashboard THEN the system SHALL support reading from the database.
3. WHEN the schema changes THEN the system SHALL manage database schema with versioned migrations.

### Requirement 14: Automated Retraining Pipeline (ROADMAP)

**User Story:** As an ML operations engineer, I want scheduled automated retraining, so that the model stays current without manual runs.

#### Acceptance Criteria

1. WHEN a configured schedule triggers THEN the system SHALL run the full training and scoring pipeline automatically via an orchestrator (e.g., Apache Airflow).
2. IF a pipeline run fails THEN the system SHALL alert operators and SHALL not overwrite the existing production model with a failed artifact.
3. WHEN a new model is trained THEN the system SHALL only promote it to production if it meets or exceeds a configured AUC-ROC threshold relative to the current model.

### Requirement 15: Customer Lifetime Value Integration (ROADMAP)

**User Story:** As a retention manager, I want churn risk weighted by customer value, so that retention spend is prioritized toward high-value accounts.

#### Acceptance Criteria

1. WHEN scoring customers THEN the system SHALL compute or ingest a Customer Lifetime Value (CLV) estimate per customer.
2. WHEN prioritizing outreach THEN the system SHALL combine churn probability and CLV into a single priority ranking (e.g., expected revenue at risk).
3. WHEN recommendations are generated THEN the system SHALL allow retention budget to be allocated according to CLV-weighted risk.

### Requirement 16: Retention A/B Testing (ROADMAP)

**User Story:** As a retention strategist, I want to test retention offers experimentally, so that I can measure which actions actually reduce churn.

#### Acceptance Criteria

1. WHEN a retention experiment is configured THEN the system SHALL assign eligible customers to control and treatment groups.
2. WHEN an experiment runs THEN the system SHALL record which offer each customer received and their subsequent churn outcome.
3. WHEN an experiment concludes THEN the system SHALL report the measured difference in churn (and confidence) between groups.
