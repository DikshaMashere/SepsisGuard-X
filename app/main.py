from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from app.schemas.prediction import PredictionRequest
from app.schemas.response import PredictionResponse
from app.schemas.trend_prediction import TrendPredictionRequest
from app.services.model_service import load_model, predict_sepsis_risk
from app.services.explainability_service import explain_sepsis_prediction
from app.services.trend_service import analyze_patient_trend
from app.services.live_uncertainty_service import estimate_live_uncertainty
from app.services.temporal_model_service import predict_temporal_risk


app = FastAPI(
    title="SepsisGuard-X API",
    description="Sepsis risk prediction research prototype",
    version="1.0.0",
)

FRONTEND_PATH = Path(__file__).resolve().parents[1] / "frontend" / "index.html"

DISCLAIMER = (
    "Research prototype only. This output is not a clinical diagnosis "
    "and must not be used to make patient-care decisions."
)

REFERENCE_THRESHOLD = 0.03


@app.get("/", include_in_schema=False)
def home():
    if not FRONTEND_PATH.is_file():
        raise HTTPException(status_code=500, detail="Frontend file not found")
    return FileResponse(FRONTEND_PATH)


@app.get("/api/health")
def health_check():
    try:
        load_model()
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_name": "Calibrated Logistic Regression",
        }
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model unavailable: {exc}",
        ) from exc


@app.post("/api/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        probability = predict_sepsis_risk(request.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the application logs.",
        ) from exc

    return PredictionResponse(
        probability=probability,
        model_name="Calibrated Logistic Regression",
        threshold=REFERENCE_THRESHOLD,
        threshold_result=(
            "At or above reference threshold"
            if probability >= REFERENCE_THRESHOLD
            else "Below reference threshold"
        ),
        disclaimer=DISCLAIMER,
    )


@app.post("/api/predict-temporal")
def predict_temporal(request: TrendPredictionRequest):
    try:
        observations = [
            observation.model_dump()
            for observation in request.observations
        ]
        result = predict_temporal_risk(observations)
        result["disclaimer"] = DISCLAIMER
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Temporal prediction failed. Check the application logs.",
        ) from exc


@app.post("/api/explain")
def explain_prediction(request: PredictionRequest):
    try:
        result = explain_sepsis_prediction(request.model_dump())
        result["model_name"] = "Calibrated Logistic Regression"
        result["disclaimer"] = DISCLAIMER
        return result

    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Explanation failed. Check the application logs.",
        ) from exc


@app.post("/api/predict-trend")
def predict_trend(request: TrendPredictionRequest):
    try:
        observations = [
            observation.model_dump()
            for observation in request.observations
        ]
        result = analyze_patient_trend(observations)
        result["disclaimer"] = DISCLAIMER
        return result

    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Trend analysis failed. Check the application logs.",
        ) from exc


@app.post("/api/uncertainty")
def predict_uncertainty(request: PredictionRequest):
    try:
        result = estimate_live_uncertainty(request.model_dump())
        result["model_name"] = "20 Unweighted Bootstrap Models"
        result["disclaimer"] = DISCLAIMER
        return result

    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Uncertainty estimation failed. Check the application logs.",
        ) from exc
