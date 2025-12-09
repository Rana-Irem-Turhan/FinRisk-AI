import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Dict
from config import API_TITLE, API_VERSION, API_DESCRIPTION
from inference import predictor

app = FastAPI(title=API_TITLE, version=API_VERSION, description=API_DESCRIPTION)

templates = Jinja2Templates(directory="src/templates")


class PredictionRequest(BaseModel):
    features: Dict[str, float]


class PredictionResponse(BaseModel):
    prediction: str
    features_used: int


class ProbabilityResponse(BaseModel):
    probabilities: Dict[str, float]
    features_used: int


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    feature_names = predictor.get_feature_names()
    # Load top 10 features from features.json
    predictor.load_model()  # Ensure features are loaded
    top_10_features = predictor.features['top_10_features']
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "features": {"all_features": feature_names, "top_10_features": top_10_features}}
    )


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": predictor._model_loaded,
        "model_ready": predictor.model is not None
    }


@app.get("/features")
async def get_features():
    return {"features": predictor.get_feature_names()}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):

    # Validate missing features
    expected = set(predictor.get_feature_names())
    incoming = set(request.features.keys())

    missing = expected - incoming
    if missing:
        raise HTTPException(400, f"Missing features: {missing}")
    prediction = predictor.predict(request.features)

    return PredictionResponse(
        prediction=prediction,
        features_used=len(request.features)
    )


@app.post("/predict_proba", response_model=ProbabilityResponse)
async def predict_proba(request: PredictionRequest):

    # Validate missing features
    expected = set(predictor.get_feature_names())
    incoming = set(request.features.keys())

    missing = expected - incoming
    if missing:
        raise HTTPException(400, f"Missing features: {missing}")
    probabilities = predictor.predict_proba(request.features)

    return ProbabilityResponse(
        probabilities=probabilities,
        features_used=len(request.features)
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
