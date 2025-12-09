# Baseline Model Documentation

## 1. Pipeline Overview
The baseline model has been upgraded to replicate a high-performing preprocessing pipeline, significantly improving upon the initial minimal baseline.

### Preprocessing
-   **Dropped Columns:** `ID`, `Customer_ID`, `Name`, `SSN`, `Credit_Score` (Target).
-   **Data Cleaning:**
    -   **Numeric Parsing:** Cleaned `Age`, `Annual_Income`, `Outstanding_Debt`, `Num_of_Delayed_Payment`, `Num_of_Loan`, `Amount_invested_monthly`, `Monthly_Balance`, `Changed_Credit_Limit` (removed `_`, `,` and handled empty strings).
    -   **Credit_History_Age:** Parsed "X Years Y Months" into total months.
-   **Imputation & Scaling (Numeric):**
    -   `SimpleImputer(strategy='median')`
    -   `StandardScaler()`
-   **Encoding (Categorical):**
    -   `SimpleImputer(strategy='most_frequent')`
    -   `OneHotEncoder(handle_unknown='ignore')`
    -   Target (`Credit_Score`): Label Encoded.

### Model
-   **Algorithm:** Logistic Regression
-   **Parameters:** `max_iter=1000`, `class_weight='balanced'`, `random_state=42`
-   **Validation:** Stratified K-Fold Cross-Validation (5 Splits).

## 2. Performance Results

| Metric | Score |
| :--- | :--- |
| **Mean Accuracy** | **0.7211** (+/- 0.0020) |
| **Mean ROC-AUC** | **0.8647** |

### Fold-by-Fold Breakdown

| Fold | Accuracy | ROC-AUC |
| :--- | :--- | :--- |
| Fold 1 | 0.7228 | 0.8660 |
| Fold 2 | 0.7238 | 0.8647 |
| Fold 3 | 0.7180 | 0.8642 |
| Fold 4 | 0.7202 | 0.8635 |
| Fold 5 | 0.7204 | 0.8648 |

## 3. Key Findings
-   **Significant Improvement:** Accuracy improved from ~62% to ~72.1% by correctly handling dirty numeric columns (Age, Annual_Income, etc.) and using a robust preprocessing pipeline.
-   **Robustness:** Stratified K-Fold CV (5 splits) ensures the results are stable with low variance (+/- 0.0020), indicating the model generalizes well.
-   **Strong Discrimination:** ROC-AUC of 0.8647 shows the model effectively distinguishes between credit score classes despite being a simple linear model.
-   **Remaining Gap:** The target performance is ~80%. The 9% gap can be closed through advanced feature engineering (e.g., customer-level aggregation, feature interactions, loan type splitting).
-   **Next Steps:** Implement advanced feature engineering with non-linear models (Random Forest, XGBoost) to leverage complex feature relationships.
