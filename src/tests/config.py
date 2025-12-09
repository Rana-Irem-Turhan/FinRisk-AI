import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, '..', 'data')
MODELS_PATH = os.path.join(BASE_DIR, 'models')

MODEL_FILENAME = 'final_model.pkl'
MODEL_PATH = os.path.join(MODELS_PATH, MODEL_FILENAME)

FEATURES_PATH = os.path.join(MODELS_PATH, 'features.json')


# API Configurations
API_TITLE = "FinRisk-AI API"       
API_VERSION = "1.0.0"
API_DESCRIPTION = (
    "Credit Score Classification service that predicts a customer's "
    "credit category (Good, Standard, Poor). Built using a complete ML "
    "pipeline and the system decided to utilize the model which uses an optimized "
    "stacked ensemble (Random Forest + XGBoost + Logistic Regression) "
    "achieving strong accuracy and robust generalization. Suitable for "
    "automated underwriting and risk assessment."
)

# Risk levels and messages (placeholders)
RISK_LEVELS = {
    'low': (0.0, 0.3),
    'medium': (0.3, 0.7),
    'high': (0.7, 1.0)
}

RISK_MESSAGES = {
    'low': 'Low risk',
    'medium': 'Medium risk',
    'high': 'High risk'
}
