# Feature Engineering Results

## 1. Implemented Strategy
We implemented a comprehensive feature engineering pipeline with customer-level aggregation:

### Data Cleaning & Type Conversion
-   **Numeric Parsing:** Cleaned `Age`, `Annual_Income`, `Outstanding_Debt`, `Num_of_Delayed_Payment`, `Num_of_Loan`, `Amount_invested_monthly`, `Monthly_Balance`, `Changed_Credit_Limit` (removed underscores, commas, and handled invalid values).

### Feature Extraction
-   **Credit History Age:** Converted from "X Years Y Months" format to total months.
-   **Loan Features:**
    -   `Loan_Count_Calculated`: Count of different loan types.
    -   `Loan_<Type>`: One-Hot encoded top 8 loan types (Auto, Credit-Builder, Personal, Home Equity, Mortgage, Student, Debt Consolidation, Payday).
-   **Financial Ratios:**
    -   `Debt_to_Income_Ratio`: Outstanding Debt / Annual Income (financial risk metric).
    -   `Debt_Per_Loan`: Outstanding Debt / Loan Count.
    -   `Installment_to_Income`: Monthly EMI / Monthly Salary (debt service capacity).
    -   `Delayed_Per_Loan`: Num of Delayed Payments / Loan Count (payment reliability).
-   **Interaction Features:**
    -   `DTI_x_LoanCount`: Debt-to-Income × Loan Count (combined risk).
    -   `Log_Annual_Income`: Log-transformed income (handles skewness).

### Imputation & Aggregation
-   **Grouped Imputation:** Median salary imputation grouped by Occupation (more accurate than global median).
-   **Customer-Level Aggregation:** Reduced 150,000 monthly rows to 25,000 unique customers:
    -   **Stable fields** (Age, loan flags): First value.
    -   **Monthly-changing fields** (Income, Balance, EMI): Mean.
    -   **Count fields** (Delayed payments, inquiries): Sum.
    -   **Categorical fields** (Payment Behaviour, Credit Mix): Mode.

### Encoding & Scaling
-   **Ordinal Encoding:** Credit_Mix (Bad=0, Standard=1, Good=2).
-   **One-Hot Encoding:** Occupation, Payment_Behaviour, Month.
-   **Label Encoding:** Target (Credit_Score).
-   **No Global Scaling:** Features remain unscaled to preserve tree model performance (trees are invariant to feature scaling).

## 2. Model Performance Comparison

| Model | Dataset | Accuracy | Notes |
| :--- | :--- | :--- | :--- |
| **Baseline** (Simple Logistic Regression) | 5-Fold CV | **0.7211** | Strong linear baseline |
| **Logistic Regression** (with feature engineering + scaling) | Validation Split | **0.6544** | Linear model struggles with complex features |
| **Random Forest** (hyperparameter tuned) | Validation Split | **0.7340** | ✅ **Exceeds baseline by 1.3%** |

### Random Forest Class-Wise Performance

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| Poor (0) | 0.59 | 0.84 | 0.69 | 501 |
| Standard (1) | 0.73 | 0.81 | 0.77 | 832 |
| Good (2) | 0.86 | 0.63 | 0.73 | 1,167 |
| **Weighted Avg** | **0.76** | **0.73** | **0.73** | **2,500** |

## 3. Key Insights

### Why Logistic Regression Performance Dropped
1. **Non-linear Feature Interactions:** Engineered features (DTI × LoanCount, Debt_Per_Loan) capture non-linear relationships that linear models cannot leverage.
2. **Dimensionality Curse:** One-Hot encoding of multiple categorical features (Occupation, Payment_Behaviour) increased feature space without linear model regularization.
3. **Information Loss:** Dropping `Annual_Income` in favor of `Log_Annual_Income` may have removed linear signal if the true relationship isn't purely logarithmic.

### Why Random Forest Excels
1. **Non-linear Decision Boundaries:** Trees naturally capture feature interactions without explicit engineering.
2. **High Recall on Poor Scores:** 84% recall on class 0 (Poor) is critical for risk management—catches risky customers.
3. **Balanced Performance:** Weighted F1-score of 0.73 shows good generalization across all credit score classes.
4. **Robustness:** Hyperparameters (max_depth=10, balanced_class_weight) prevent overfitting while leveraging complex features.

## 4. Recommendations for Next Phase (04_model_optimization.ipynb)

✅ **Keep the engineered features** — They provide valuable signal for non-linear models.
✅ **Continue with tree-based models** — Random Forest, XGBoost will unlock feature complexity better than linear models.
✅ **Perform feature importance analysis** — Identify which engineered features drive predictions.
✅ **Cross-validate with stratified k-fold** — Ensure 73.4% accuracy is stable across data splits.
✅ **Compare with XGBoost** — Gradient boosting may outperform bagging approaches.
✅ **Class-wise optimization** — Focus on improving recall for "Poor" customers (high-risk detection).

## 5. Data Quality Improvements Made
-   ✅ Handled missing values in `Monthly_Inhand_Salary`, `Type_of_Loan`, `Credit_History_Age`.
-   ✅ Cleaned numeric columns with special characters (underscores, commas).
-   ✅ Removed outliers in `Num_of_Delayed_Payment` (clipped at 99th percentile).
-   ✅ Aggregated to customer level to prevent temporal leakage and reduce noise.
-   ✅ Verified no NaN values remain before model training.
