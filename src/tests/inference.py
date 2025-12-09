import json
import joblib
import pandas as pd
from typing import Dict
from config import MODEL_PATH, FEATURES_PATH


class CreditScorePredictor:
    def __init__(self):
        self.model = None
        self.features = None
        self._model_loaded = False

    def load_model(self):
        if not self._model_loaded:
            self.model = joblib.load(MODEL_PATH)
            with open(FEATURES_PATH, 'r') as f:
                self.features = json.load(f)
            self._model_loaded = True

    def predict(self, features_dict: Dict[str, float]) -> str:
        # Ensure model is loaded
        self.load_model()

        df = pd.DataFrame([features_dict])

        # Ensure correct feature order
        df = df[
            self.features['all_features']
        ]

        # Get prediction
        pred_class = self.model.predict(df)[0]

        # Map to credit score labels
        credit_labels = {0: 'Poor', 1: 'Standard', 2: 'Good'}
        prediction = credit_labels.get(pred_class, 'Unknown')

        return prediction

    def predict_proba(self, features_dict: Dict[str, float]) -> Dict[str, float]:
        # Ensure model is loaded
        self.load_model()

        df = pd.DataFrame([features_dict])

        # Ensure correct feature order
        df = df[self.features['all_features']]

        # Get prediction probabilities
        proba = self.model.predict_proba(df)[0]

        # Map to credit score labels
        credit_labels = {0: 'Poor', 1: 'Standard', 2: 'Good'}
        probabilities = {
            credit_labels[i]: float(proba[i]) for i in range(len(proba))
        }

        return probabilities

    def get_feature_names(self):
        # Ensure model is loaded to get feature names
        self.load_model()
        return self.features['all_features']

    def get_top_features(self, n=10):
        # Ensure model is loaded
        self.load_model()
        # Top 10 most important features based on model evaluation
        top_features = [
            'Credit_Mix_Ordinal',
            'Outstanding_Debt',
            'Delay_from_due_date',
            'Payment_of_Min_Amount_Yes',
            'Changed_Credit_Limit',
            'Credit_Utilization_Ratio',
            'Monthly_Balance',
            'Num_Bank_Accounts',
            'Num_Credit_Inquiries',
            'Annual_Income'
        ]
        return top_features[:n]


predictor = CreditScorePredictor()
