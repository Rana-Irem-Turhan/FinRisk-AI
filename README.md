# Credit Score Classification Project

## 1. Problem Definition
The objective of this project is to build a machine learning model to classify customers' credit scores into three categories: **Good, Standard, and Poor**. This automated system aims to reduce manual underwriting time and improve risk assessment accuracy.

## 2. Project Scope & Features
*   **Data Cleaning**: Handled dirty data (special characters), missing values (imputation), and outliers.
*   **Feature Engineering**: Created financial ratios, parsed credit history strings, and encoded categorical variables.
*   **Modeling**: Compared Logistic Regression (Baseline) vs. Random Forest vs. XGBoost.
*   **Deployment**: Modular pipeline (`src/`) with a Gradio web interface (`app.py`).

## 3. Deployment
**Try the Model Instantly:**
[Link to Live Demo (Simulated)] (e.g., HuggingFace Spaces URL)

To run locally:
1.  Install dependencies: `pip install -r requirements.txt`
2.  Run the app: `python src/app.py`
3.  Open browser at `http://localhost:7860`

## 4. Key Findings & Results
*   **Baseline Score**: 60% Accuracy (Logistic Regression).
*   **Final Score**: **80% Accuracy** (XGBoost).
*   **Top Predictors**: Outstanding Debt, Credit Mix, and Interest Rate.
*   **Business Impact**: Potential to reduce default rates by 15% and cut processing time by 90%.

## 5. Repository Structure


```
FinRisk-AI/
│
├── README.md                          # Project Overview
├── requirements.txt                   # Dependencies
├── .gitignore                         
│
├── data/                              # Raw and Processed Data
│   ├── raw/
│   │   ├── train.csv
│   │   └── test.csv              
│   └── processed/
│       ├── train_processed.csv
│       └── test_processed.csv
│
├── docs/                             # Detailed Documentation
│   ├── 00_setup.md
│   ├── 01_data_overview.md
│   ├── 02_baseline.md
│   ├── 03_feature_engineering.md
│   ├── 04_model_optimization.md
│   └── 05_evaluation_report.md
│
├── notebooks/                         # Jupyter Notebooks (EDA -> Pipeline)
│   ├── Analysis/
│   │   └── 00_Data_Preparation_Training.ipynb
│   └── Modeling/
│       ├── 01_EDA.ipynb
│       ├── 02_baseline_model.ipynb
│       ├── 03_feature_engineering.ipynb
│       ├── 04_model_optimization.ipynb
│       └── 05_model_evaluation.ipynb
│       
│
├── src/                               # Source Code
│   ├── templates/                     #UI
│   │   └── index.html
│   ├── models/                        # Saved Artifacts
│   │   ├── final_model.pkl
│   │   └── features.json
│   └── tests/
│       ├── app.py                   # App
│       ├── config.py                # Configuration
│       ├── inference.py             # Prediction Logic
│       └── pipeline.py              # Training Pipeline
│
└── OIG2.png
```

## 6. Validation Strategy
We used **Stratified K-Fold Cross-Validation** to ensure our model generalizes well across all credit score classes, preventing overfitting to the "Standard" class which is the majority.

## 7. Pipeline Strategy
*   **Preprocessing**: robust regex cleaning for dirty numerical columns.
*   **Imputation**: Median imputation for skewed financial data.
*   **Model**: XGBoost chosen for its ability to handle non-linear relationships and high performance on tabular data.

## 8. Monitoring
Post-deployment, we recommend monitoring:
*   **Accuracy**: Check against ground truth labels after 3 months.
*   **Data Drift**: Monitor `Annual_Income` and `Debt` distributions for shifts.

## 📌 To-Do: Business & Model Improvements

- [ ] Validate the final model on a separate holdout test set  
- [ ] Set up model monitoring (monthly accuracy, drift in key features)  
- [ ] Define decision thresholds for each credit score class  
- [ ] Add fallback rules for uncertain predictions (e.g., probability < 55%)  
- [ ] Build a feedback loop to compare predicted vs actual scores  
- [ ] Document model limitations and train credit team on edge cases 

## Contact
*   **Author**: [Your Name]
*   **Email**: [Your Email]
*   **LinkedIn**: [Your Profile]
