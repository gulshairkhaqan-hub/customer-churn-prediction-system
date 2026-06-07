# Customer Churn Prediction System

[Python 3.10+] [XGBoost] [LightGBM] [SHAP] [Streamlit] [Teyzix Core ML-3]

## 1. Project Overview

This project is an end-to-end machine learning system that predicts which telecom customers are most likely to cancel their service. It is built on the IBM Telco Customer Churn dataset and combines model training, SHAP-based explanations, weekly batch scoring, and a Streamlit dashboard. The goal is to give a retention team a ranked list of at-risk customers and a concrete next action for each one, instead of generic "all customers are equal" outreach.

## 2. Business Problem

Telecom companies lose recurring revenue every time a customer leaves, and acquiring a new customer typically costs five to seven times more than keeping an existing one. The retention team cannot call every customer every month, so they need a way to focus on the accounts most likely to churn next. This system identifies at-risk customers before they churn, explains why each one is at risk, and proposes a retention action so the team can act with confidence and a limited budget.

## 3. Dataset Description

- **Source**: IBM Telco Customer Churn Dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)
- **Rows**: 7,043 customers
- **Features**: 20 input features + 1 target column (`Churn`)
- **Target**: `Churn` — `Yes` (encoded as 1) or `No` (encoded as 0)
- **Class balance**: ~26.5% churned, ~73.5% stayed (imbalanced)

### Columns

| Column | Type | Description |
| --- | --- | --- |
| `customerID` | string | Unique customer identifier (dropped before training) |
| `gender` | category | Customer gender (Female / Male) |
| `SeniorCitizen` | int | 1 if customer is a senior citizen, else 0 |
| `Partner` | category | Whether the customer has a partner (Yes / No) |
| `Dependents` | category | Whether the customer has dependents (Yes / No) |
| `tenure` | int | Months the customer has stayed with the company |
| `PhoneService` | category | Whether the customer has phone service (Yes / No) |
| `MultipleLines` | category | Multiple phone lines (Yes / No / No phone service) |
| `InternetService` | category | Internet type (DSL / Fiber optic / No) |
| `OnlineSecurity` | category | Online security add-on (Yes / No / No internet service) |
| `OnlineBackup` | category | Online backup add-on (Yes / No / No internet service) |
| `DeviceProtection` | category | Device protection plan (Yes / No / No internet service) |
| `TechSupport` | category | Tech support add-on (Yes / No / No internet service) |
| `StreamingTV` | category | Streaming TV add-on (Yes / No / No internet service) |
| `StreamingMovies` | category | Streaming movies add-on (Yes / No / No internet service) |
| `Contract` | category | Contract term (Month-to-month / One year / Two year) |
| `PaperlessBilling` | category | Paperless billing flag (Yes / No) |
| `PaymentMethod` | category | Payment method (Electronic check / Mailed check / Bank transfer / Credit card) |
| `MonthlyCharges` | float | Amount charged to the customer per month |
| `TotalCharges` | float | Total amount charged over the customer's lifetime |
| `Churn` | category | Target — whether the customer left (Yes / No) |

> Note: `TotalCharges` arrives as text in the raw file because some new customers have a blank value (`" "`). The preprocessing step coerces it to numeric and median-fills the missing rows.

## 4. Feature Engineering

Seven business-driven features are added before encoding:

1. **`tenure_group`** — Lifecycle stage encoded 0–3 (New / Early / Mid / Loyal). Captures the well-known fact that new customers churn far more than long-tenured ones.
2. **`charges_per_month_ratio`** — `TotalCharges / (tenure + 1)`. Detects sudden billing spikes that often precede churn.
3. **`has_multiple_services`** — Count of add-on services the customer holds (0–6). More services = higher switching cost = lower churn.
4. **`is_high_value`** — `1` if `MonthlyCharges > 70`. High-paying customers are valuable but more demanding.
5. **`contract_risk`** — Ordinal score: Month-to-month=3, One year=2, Two year=1. Month-to-month customers can leave any cycle.
6. **`payment_risk`** — Ordinal score: Electronic check=2, Mailed check=1, Bank transfer / Credit card=0. Auto-pay reduces churn friction.
7. **`no_support_services`** — `1` if both `TechSupport` and `OnlineSecurity` are `No`. Customers without support feel unsupported and leave.

## 5. ML Models Trained

| Model | Purpose | Class-imbalance handling |
| --- | --- | --- |
| Logistic Regression | Simple baseline that sets the floor every other model has to beat | `class_weight="balanced"` |
| XGBoost | Primary gradient-boosted trees model | `scale_pos_weight = neg/pos` |
| LightGBM | Faster gradient boosting for comparison | `class_weight="balanced"` |

**Evaluation metrics**: AUC-ROC (primary), Precision, Recall (most important for revenue protection), F1, Accuracy. The model with the highest test-set AUC-ROC is selected as `best_model.pkl`.

## 6. SHAP Explainability

SHAP (SHapley Additive exPlanations) explains each prediction by attributing how much each feature pushed the predicted churn probability up or down. We use it both globally (which features matter most across all customers) and locally (the top 3 reasons a specific customer is at risk). This turns a "black box" model into something a retention analyst can actually defend in a meeting.

## 7. Risk Segmentation Logic

The weekly scoring pipeline assigns every customer to one of three tiers based on predicted churn probability:

| Tier | Probability range | Suggested response |
| --- | --- | --- |
| **HIGH** | ≥ 0.70 | Call within 48 hours |
| **MEDIUM** | 0.40 – 0.69 | Add to nurture campaign, watch list |
| **LOW** | < 0.40 | No action, routine engagement |

## 8. Retention Recommendation Examples

The recommender maps each HIGH-risk customer's top SHAP feature to a concrete action. Examples:

1. Top reason is **`contract_risk`** (month-to-month) → *Offer annual contract with 20% discount.*
2. Top reason is **`MonthlyCharges`** (high bill) → *Offer discounted plan or loyalty reward.*
3. Top reason is **`no_support_services`** → *Offer free tech support upgrade for 3 months.*

Other rules cover low tenure, fiber-optic complaints, and a default "schedule retention call within 48 hours" for anything else.

## 9. Project Structure

```
customer-churn-prediction/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
│   └── churn_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── feature_engineer.py
│   ├── model_trainer.py
│   ├── explainer.py
│   ├── scorer.py
│   ├── recommender.py
│   └── visualizer.py
├── models/                # Saved .pkl artifacts
├── outputs/               # Generated CSVs and PNGs
├── reports/               # Markdown reports
├── app.py                 # Streamlit dashboard
├── main.py                # Pipeline orchestrator
├── requirements.txt
└── README.md
```

## 10. Tech Stack

- **Language**: Python 3.10+
- **Data**: pandas, numpy
- **Modeling**: scikit-learn, XGBoost, LightGBM
- **Explainability**: SHAP
- **Visualization**: matplotlib, seaborn, plotly
- **App**: Streamlit
- **Persistence**: joblib

## 11. Installation

```bash
git clone <repo-url>
cd customer-churn-prediction
pip install -r requirements.txt
```

Place the dataset at `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` before running the pipeline.

## 12. How to Run

Run the full training and scoring pipeline (loads data, trains models, generates SHAP explanations, scores customers, builds recommendations, and saves all artifacts):

```bash
python main.py
```

Launch the interactive dashboard:

```bash
streamlit run app.py
```

On Windows PowerShell, if `streamlit` is not on PATH, use:

```bash
python -m streamlit run app.py
```

## 13. Model Evaluation Results
Test-set performance of the best model (Logistic Regression):

| Metric | Score |
| --- | --- |
| AUC-ROC | 0.8448 |
| F1 | 0.6217 |
| Recall | 0.7888 |
| Precision | 0.5130 |
## 14. Key Business Insights

## 14. Key Business Insights

1. 353 out of 1,409 customers (25%) are HIGH risk — these need retention action within 48 hours.
2. Top churn drivers: Fiber optic internet, short tenure, and month-to-month contracts — these three appear in nearly every HIGH-risk customer's top reasons.
3. Logistic Regression outperformed XGBoost and LightGBM on AUC-ROC (0.8448) — simpler model won on this dataset, suggesting churn patterns are largely linear.

## 15. Future Improvements

- Real-time scoring API with FastAPI
- PostgreSQL backend for production data storage
- Automated retraining pipeline with Apache Airflow
- Customer Lifetime Value (CLV) integration so retention spend matches account value
- A/B testing framework to measure which retention offers actually work

## 16. Author

**Gul Shair** | Teyzix Core Internship | Task ML-3
