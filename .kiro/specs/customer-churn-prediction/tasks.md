# Implementation Plan: Customer Churn Prediction System

## Overview

This plan converts the design into incremental, test-backed coding steps for the as-is system (Requirements 1–11), implemented in **Python**. Each pipeline stage in `src/` is built (or verified) against its design contract and immediately backed by tests. Correctness properties from the design are turned into Hypothesis property-based tests, each as its own sub-task annotated with its property number and the requirement clause it validates. Example, edge-case, integration, and smoke tests cover behavior that is not expressed as a universal property (reporting, plot side effects, persistence, orchestration ordering, dashboard read-only behavior, and config presence).

Roadmap Requirements 12–16 are intentionally excluded — they are not implemented and have no properties yet.

## Tasks

- [ ] 1. Set up test infrastructure and shared generators
  - [ ] 1.1 Configure pytest + Hypothesis and build Telco-shaped strategies
    - Add `pytest` and `hypothesis` to the dev/test setup and create a `tests/` package
    - Create `tests/strategies.py` with Hypothesis strategies that generate realistic Telco-shaped DataFrames: random `tenure`, `MonthlyCharges`, `TotalCharges` (including blank/space strings), valid `Contract`/`PaymentMethod`/service categories, and `Yes`/`No` targets
    - Include boundary values (tenure 12/13/24/25/48/49, MonthlyCharges 70, probabilities 0.40/0.70) and a synthetic-SHAP-array strategy (2D, list-of-arrays, 3D, `Explanation`) for SHAP properties
    - Configure `@settings(max_examples=100)` as the project default
    - _Requirements: 11.1, 11.2_

- [ ] 2. Implement data loading and validation (`src/data_loader.py`)
  - [ ] 2.1 Implement/verify `load_and_analyze(filepath)`
    - Load the CSV into a DataFrame; report shape, dtypes, head, per-column missing values, duplicate count
    - Detect and report blank/whitespace-only `TotalCharges` cells; report churn counts and churn rate percentage
    - Return the raw (uncleaned) DataFrame
    - _Requirements: 1.1, 1.3, 1.4, 1.5, 1.6_

  - [ ]* 2.2 Write property test for blank/whitespace TotalCharges detection
    - **Property 1: Blank/whitespace TotalCharges detection**
    - **Validates: Requirements 1.5**
    - File: `tests/properties/test_property_01_blank_totalcharges.py`

  - [ ]* 2.3 Write property test for churn rate computation
    - **Property 2: Churn rate computation**
    - **Validates: Requirements 1.6**
    - File: `tests/properties/test_property_02_churn_rate.py`

  - [ ]* 2.4 Write example/edge tests for data loading
    - Example: load + report against the real dataset (1.1, 1.3, 1.4)
    - Edge case: nonexistent path raises a clear `FileNotFoundError` and stops the pipeline (1.2)
    - File: `tests/test_data_loader.py`
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [ ] 3. Implement feature engineering (`src/feature_engineer.py`)
  - [ ] 3.1 Implement/verify `engineer_features(df)`
    - Operate on raw string categoricals; coerce `TotalCharges` to `float64`
    - Add `tenure_group`, `charges_per_month_ratio`, `has_multiple_services`, `is_high_value`, `contract_risk`, `payment_risk`, `no_support_services`
    - Print a per-feature summary and return the augmented DataFrame
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10_

  - [ ]* 3.2 Write property test for TotalCharges float64 coercion
    - **Property 3: TotalCharges coerced to float64**
    - **Validates: Requirements 2.2**
    - File: `tests/properties/test_property_03_totalcharges_float.py`

  - [ ]* 3.3 Write property test for tenure_group banding
    - **Property 4: tenure_group ordinal banding**
    - **Validates: Requirements 2.3**
    - File: `tests/properties/test_property_04_tenure_group.py`

  - [ ]* 3.4 Write property test for charges_per_month_ratio identity
    - **Property 5: charges_per_month_ratio identity**
    - **Validates: Requirements 2.4**
    - File: `tests/properties/test_property_05_charges_ratio.py`

  - [ ]* 3.5 Write property test for has_multiple_services count
    - **Property 6: has_multiple_services count**
    - **Validates: Requirements 2.5**
    - File: `tests/properties/test_property_06_multiple_services.py`

  - [ ]* 3.6 Write property test for is_high_value threshold
    - **Property 7: is_high_value threshold**
    - **Validates: Requirements 2.6**
    - File: `tests/properties/test_property_07_is_high_value.py`

  - [ ]* 3.7 Write property test for contract_risk mapping
    - **Property 8: contract_risk mapping**
    - **Validates: Requirements 2.7**
    - File: `tests/properties/test_property_08_contract_risk.py`

  - [ ]* 3.8 Write property test for payment_risk mapping
    - **Property 9: payment_risk mapping**
    - **Validates: Requirements 2.8**
    - File: `tests/properties/test_property_09_payment_risk.py`

  - [ ]* 3.9 Write property test for no_support_services indicator
    - **Property 10: no_support_services indicator**
    - **Validates: Requirements 2.9**
    - File: `tests/properties/test_property_10_no_support_services.py`

  - [ ]* 3.10 Write property test for presence of all engineered features
    - **Property 11: All engineered features present**
    - **Validates: Requirements 2.10**
    - File: `tests/properties/test_property_11_all_features_present.py`

- [ ] 4. Implement data preprocessing (`src/preprocessor.py`)
  - [ ] 4.1 Implement/verify `preprocess_data(df)`
    - Clean blank/whitespace `TotalCharges` → median fill; drop `customerID`; encode `Churn` to 1/0
    - Label-encode binary columns; one-hot encode multi-class with `drop_first=True`, excluding/dropping `Contract` and `PaymentMethod`; cast bool dummies to int
    - Stratified 80/20 split (`random_state=42`); fit `StandardScaler` on `[tenure, MonthlyCharges, TotalCharges]` from train only and apply to both splits
    - Return `X_train, X_test, y_train, y_test, scaler, feature_names, df_processed`
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9_

  - [ ]* 4.2 Write property test for TotalCharges cleaning
    - **Property 12: TotalCharges cleaning leaves no missing values**
    - **Validates: Requirements 3.1**
    - File: `tests/properties/test_property_12_totalcharges_clean.py`

  - [ ]* 4.3 Write property test for clean numeric matrix
    - **Property 13: Preprocessing produces a clean numeric matrix**
    - **Validates: Requirements 3.2, 3.5, 3.6**
    - File: `tests/properties/test_property_13_clean_matrix.py`

  - [ ]* 4.4 Write property test for Churn target encoding
    - **Property 14: Churn target encoding**
    - **Validates: Requirements 3.3**
    - File: `tests/properties/test_property_14_churn_encoding.py`

  - [ ]* 4.5 Write property test for binary column label encoding
    - **Property 15: Binary column label encoding**
    - **Validates: Requirements 3.4**
    - File: `tests/properties/test_property_15_binary_encoding.py`

  - [ ]* 4.6 Write property test for stratified, reproducible split
    - **Property 16: Stratified, reproducible train/test split**
    - **Validates: Requirements 3.7, 11.2**
    - File: `tests/properties/test_property_16_split.py`

  - [ ]* 4.7 Write property test for scaler fitted on training data only
    - **Property 17: Scaler fitted on training data only**
    - **Validates: Requirements 3.8**
    - File: `tests/properties/test_property_17_scaler.py`

- [ ] 5. Checkpoint - data pipeline
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. Implement model training and selection (`src/model_trainer.py`)
  - [ ] 6.1 Implement/verify `train_and_compare(...)`
    - Scrub residual NaN to 0 preserving column names; train Logistic Regression, XGBoost, LightGBM with imbalance handling
    - Compute AUC-ROC/Precision/Recall/F1/Accuracy on test set; print comparison table and per-model confusion matrices
    - Select best model by test AUC-ROC; save the three models plus `best_model.pkl`
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_

  - [ ]* 6.2 Write property test for NaN scrub preserving columns
    - **Property 18: NaN scrub preserves feature columns**
    - **Validates: Requirements 4.3**
    - File: `tests/properties/test_property_18_nan_scrub.py`

  - [ ]* 6.3 Write property test for best-model AUC-ROC argmax
    - **Property 19: Best model is the AUC-ROC argmax**
    - **Validates: Requirements 4.6**
    - File: `tests/properties/test_property_19_best_model.py`

  - [ ]* 6.4 Write example tests for model training
    - Three models trained, imbalance config applied, metric keys present and in [0, 1]
    - File: `tests/test_model_trainer.py`
    - _Requirements: 4.1, 4.2, 4.4, 4.5_

- [ ] 7. Implement explainability (`src/explainer.py`)
  - [ ] 7.1 Implement/verify SHAP functions
    - `generate_shap_values` selects Tree/Linear explainer with generic fallback and normalizes output to a 2D positive-class array
    - `plot_shap_summary` saves top-15 summary to `outputs/shap_summary.png`; `explain_single_customer` returns top-3 features with direction; `generate_explanation_text` renders plain language
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

  - [ ]* 7.2 Write property test for SHAP normalization
    - **Property 20: SHAP normalization to 2D positive-class array**
    - **Validates: Requirements 5.3**
    - File: `tests/properties/test_property_20_shap_normalize.py`

  - [ ]* 7.3 Write property test for single-customer top-3 explanation
    - **Property 21: Single-customer top-3 explanation**
    - **Validates: Requirements 5.5**
    - File: `tests/properties/test_property_21_single_customer.py`

  - [ ]* 7.4 Write property test for explanation text referencing each reason
    - **Property 22: Explanation text references each top reason**
    - **Validates: Requirements 5.6**
    - File: `tests/properties/test_property_22_explanation_text.py`

  - [ ]* 7.5 Write example/edge tests for explainer
    - Example: correct explainer per model type; summary PNG saved (5.1, 5.4)
    - Edge case: forced primary-explainer failure falls back without crashing (5.2)
    - File: `tests/test_explainer.py`
    - _Requirements: 5.1, 5.2, 5.4_

- [ ] 8. Checkpoint - modeling and explainability
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 9. Implement weekly scoring and risk segmentation (`src/scorer.py`)
  - [ ] 9.1 Implement/verify `run_weekly_scoring(...)`
    - Fill NaN with 0; compute positive-class probability per customer; assign HIGH/MEDIUM/LOW tiers
    - Build scoring table (`customer_index`, `churn_probability` rounded to 4 dp, `risk_level`, `scoring_week`, `scoring_date`); print per-tier counts/percentages; save `outputs/weekly_scoring_week_{n}.csv`
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

  - [ ]* 9.2 Write property test for valid churn probabilities
    - **Property 23: Churn probabilities are valid**
    - **Validates: Requirements 6.1**
    - File: `tests/properties/test_property_23_probabilities.py`

  - [ ]* 9.3 Write property test for risk tiering thresholds
    - **Property 24: Risk tiering thresholds**
    - **Validates: Requirements 6.3**
    - File: `tests/properties/test_property_24_risk_tiers.py`

  - [ ]* 9.4 Write property test for scoring table schema and rounding
    - **Property 25: Scoring table schema and rounding**
    - **Validates: Requirements 6.4**
    - File: `tests/properties/test_property_25_scoring_schema.py`

  - [ ]* 9.5 Write example test for per-tier scoring summary
    - Per-tier counts/percentages printed; CSV written for week number (6.2, 6.5, 6.6)
    - File: `tests/test_scorer.py`
    - _Requirements: 6.2, 6.5, 6.6_

- [ ] 10. Implement retention recommendations (`src/recommender.py`)
  - [ ] 10.1 Implement/verify `generate_recommendations(...)`
    - Filter to HIGH-risk customers; select top-3 risk-increasing SHAP drivers (fallback to absolute magnitude); map top driver to action via priority-ordered rules
    - Produce table with `customer_index`, `risk_level`, `churn_probability`, `top_reason_1/2/3`, `recommended_action`; write `outputs/retention_recommendations.csv`
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.6_

  - [ ]* 10.2 Write property test for recommendation coverage of HIGH-risk customers
    - **Property 26: Recommendations cover exactly the HIGH-risk customers**
    - **Validates: Requirements 7.1, 7.4**
    - File: `tests/properties/test_property_26_rec_coverage.py`

  - [ ]* 10.3 Write property test for top-3 risk-increasing drivers
    - **Property 27: Top-3 risk-increasing drivers**
    - **Validates: Requirements 7.2**
    - File: `tests/properties/test_property_27_top_drivers.py`

  - [ ]* 10.4 Write property test for rule-based action mapping
    - **Property 28: Rule-based action mapping**
    - **Validates: Requirements 7.3**
    - File: `tests/properties/test_property_28_action_mapping.py`

  - [ ]* 10.5 Write edge-case test for no HIGH-risk customers
    - Header-only CSV written and empty DataFrame returned without error (7.5)
    - File: `tests/test_recommender.py`
    - _Requirements: 7.5_

- [ ] 11. Implement visualizations (`src/visualizer.py`)
  - [ ] 11.1 Implement/verify plotting functions
    - EDA charts on original categorical values, model comparison + per-model confusion matrices, risk segmentation, and churn probability distribution with 0.40/0.70 markers
    - Each function saves a PNG to `outputs/` and returns the path; confusion-matrix filename is filesystem-safe
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [ ]* 11.2 Write property test for filesystem-safe confusion-matrix filename
    - **Property 29: Filesystem-safe confusion-matrix filename**
    - **Validates: Requirements 8.5**
    - File: `tests/properties/test_property_29_cm_filename.py`

  - [ ]* 11.3 Write example tests for visualization outputs
    - Each plot writes its expected PNG to `outputs/` (8.1–8.4)
    - File: `tests/test_visualizer.py`
    - _Requirements: 8.1, 8.2, 8.3, 8.4_

- [ ] 12. Checkpoint - scoring, recommendations, visualizations
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 13. Implement pipeline orchestration (`main.py`)
  - [ ] 13.1 Implement/verify `run_pipeline()`
    - Execute all stages in order with labeled section headers; anchor paths to the script directory; create directories; per-step error handling with non-zero exit on failure
    - Persist `best_model.pkl`, `scaler.pkl`, `results_dict.pkl`, `feature_names.pkl`; print completion message directing the user to the dashboard
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

  - [ ]* 13.2 Write integration test for ordered end-to-end run
    - End-to-end run produces all artifacts; identical artifacts across different working directories (9.1, 9.4)
    - File: `tests/test_pipeline_integration.py`
    - _Requirements: 9.1, 9.4_

  - [ ]* 13.3 Write example/edge tests for orchestration behavior
    - Section headers printed; a forced stage failure stops the pipeline with `SystemExit(1)` (9.2, 9.3)
    - File: `tests/test_pipeline_orchestration.py`
    - _Requirements: 9.2, 9.3_

- [ ] 14. Implement interactive dashboard (`app.py`)
  - [ ] 14.1 Implement/verify cached loaders and six views
    - Cached read-only loaders for scoring/recommendations/raw data/models plus metric and SHAP recomputation; Plotly chart builders
    - Top-nav dispatch across Overview, Model Performance, Risk Segmentation, Feature Importance & SHAP, Retention Recommendations, Customer Drill-Down; overview metrics, risk filter/sort, and drill-down details
    - _Requirements: 10.1, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8_

  - [ ]* 14.2 Write example tests for dashboard helpers
    - Loaders are read-only (never retrain), six views registered, overview metrics computed, filter/sort and drill-down helpers return expected results (10.1, 10.3, 10.4, 10.5, 10.6)
    - File: `tests/test_app.py`
    - _Requirements: 10.1, 10.3, 10.4, 10.5, 10.6_

  - [ ]* 14.3 Write edge-case test for graceful degradation
    - Each missing artifact makes its loader return `None` and the page degrades with a warning/error instead of crashing (10.2)
    - File: `tests/test_app_degradation.py`
    - _Requirements: 10.2_

- [ ] 15. Environment, configuration, and documentation
  - [ ] 15.1 Verify dependencies, theme config, and README
    - Ensure `requirements.txt` declares all dependencies, `.streamlit/config.toml` provides a dark theme, and `README.md` documents dataset, features, models, risk logic, structure, install, and run commands
    - _Requirements: 11.1, 11.3, 11.4_

  - [ ]* 15.2 Write smoke tests for artifacts, config, and docs
    - Required artifacts/files/config/docs exist and load (4.7, 6.6, 7.6, 9.5, 11.1, 11.3, 11.4)
    - File: `tests/test_smoke_artifacts.py`
    - _Requirements: 4.7, 6.6, 7.6, 9.5, 11.1, 11.3, 11.4_

- [ ] 16. Final checkpoint - full suite green
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional test sub-tasks and can be skipped for a faster MVP; core implementation tasks are never optional.
- Each property test is implemented as a single Hypothesis test, runs for at least 100 examples (`@settings(max_examples=100)`), and carries a comment in the form `Feature: customer-churn-prediction, Property {number}: {property_text}`.
- For SHAP-dependent properties (20, 21, 27), generate synthetic SHAP-shaped arrays rather than training real models, keeping iterations fast and deterministic.
- Each property test lives in its own file under `tests/properties/` so they can run independently and in parallel.
- Roadmap Requirements 12–16 are not implemented and are intentionally excluded from this plan.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["2.1", "3.1", "4.1", "6.1", "7.1", "9.1", "10.1", "11.1", "13.1", "14.1", "15.1"] },
    { "id": 2, "tasks": ["2.2", "2.3", "2.4", "3.2", "3.3", "3.4", "3.5", "3.6", "3.7", "3.8", "3.9", "3.10", "4.2", "4.3", "4.4", "4.5", "4.6", "4.7", "6.2", "6.3", "6.4", "7.2", "7.3", "7.4", "7.5", "9.2", "9.3", "9.4", "9.5", "10.2", "10.3", "10.4", "10.5", "11.2", "11.3", "13.2", "13.3", "14.2", "14.3", "15.2"] }
  ]
}
```
