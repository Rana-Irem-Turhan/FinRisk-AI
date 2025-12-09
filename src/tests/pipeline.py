import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib
import os
from config import DATA_PATH, MODELS_PATH, MODEL_FILENAME, FEATURES_PATH

# Top 10 features for user input
SELECTED_FEATURES = [
    'Credit_Mix_Ordinal',
    'Outstanding_Debt',
    'Delay_from_due_date',
    'Payment_of_Min_Amount_Yes',
    'Num_Credit_Card',
    'Interest_Rate',
    'Num_of_Delayed_Payment',
    'Installment_to_Income',
    'Num_Bank_Accounts',
    'Num_Credit_Inquiries'
]

def run_pipeline():
    print("Starting pipeline...")

    # Load processed training data
    train_processed_path = os.path.join(DATA_PATH, 'processed', 'train_processed.csv')
    if not os.path.exists(train_processed_path):
        raise FileNotFoundError(f"Processed training data not found at {train_processed_path}")

    train_processed = pd.read_csv(train_processed_path)

    # Train model with ALL FEATURES except the target
    target = 'Credit_Score'
    if target not in train_processed.columns:
        raise ValueError("Target column 'Credit_Score' is missing from processed training data.")

    X = train_processed.drop(target, axis=1)
    y = train_processed[target]

    ALL_FEATURES = X.columns.tolist()

    print(f"Training model using ALL {len(ALL_FEATURES)} features...")
    print(f"Training data loaded: {X.shape[0]} samples, {X.shape[1]} features")

    # Define models
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight='balanced',
        criterion='entropy',
        random_state=1907,
        n_jobs=-1
    )

    xgb_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        random_state=1907,
        verbosity=0
    )

    stacking_clf = StackingClassifier(
        estimators=[('rf', rf_model), ('xgb', xgb_model)],
        final_estimator=LogisticRegression(max_iter=1000, random_state=1907),
        cv=5
    )

    # Train model
    print("Training stacking classifier...")
    stacking_clf.fit(X, y)

    # Save model
    os.makedirs(MODELS_PATH, exist_ok=True)
    model_path = os.path.join(MODELS_PATH, MODEL_FILENAME)
    joblib.dump(stacking_clf, model_path)
    print(f"Model saved to {model_path}")

    # Save BOTH feature lists
    feature_data = {
        "all_features": ALL_FEATURES,
        "top_10_features": SELECTED_FEATURES
    }

    with open(FEATURES_PATH, 'w') as f:
        json.dump(feature_data, f, indent=4)

    print("Feature lists saved.")
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()
