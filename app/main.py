from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    DSO1BatchRequest,
    DSO1PredictRequest,
    DSO1PredictionResponse,
    DSO2BatchRequest,
    DSO2PredictRequest,
    DSO2PredictionResponse,
    DSO3BatchRequest,
    DSO3OverviewResponse,
    DSO3PredictRequest,
    DSO3PredictionResponse,
    EvaluationResponse,
)
from .service import DATA_PATH, ModelService


app = FastAPI(title="DSO Finance API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


try:
    service = ModelService()
except Exception as exc:  # pragma: no cover - startup guard
    service = None
    startup_error = exc
else:
    startup_error = None


def _require_service() -> ModelService:
    if service is None:
        raise HTTPException(status_code=500, detail=f"Model service unavailable: {startup_error}")
    return service


@app.get("/health")
def health() -> dict[str, object]:
    svc = _require_service()
    return {
        "status": "ok",
        "data_path": str(DATA_PATH),
        "rows": int(len(svc.raw_df)),
        "columns": list(svc.raw_df.columns),
    }


@app.post("/dso1/predict", response_model=DSO1PredictionResponse)
def predict_dso1(payload: DSO1PredictRequest) -> dict:
    return _require_service().predict_dso1(payload)


@app.post("/dso2/predict", response_model=DSO2PredictionResponse)
def predict_dso2(payload: DSO2PredictRequest) -> dict:
    return _require_service().predict_dso2(payload)


@app.post("/dso3/predict", response_model=DSO3PredictionResponse)
def predict_dso3(payload: DSO3PredictRequest) -> dict:
    return _require_service().predict_dso3(payload)


@app.post("/dso1/predict/batch")
def predict_dso1_batch(payload: DSO1BatchRequest) -> list[dict]:
    return _require_service().batch_predict_dso1(payload.rows)


@app.post("/dso2/predict/batch")
def predict_dso2_batch(payload: DSO2BatchRequest) -> list[dict]:
    return _require_service().batch_predict_dso2(payload.rows)


@app.post("/dso3/predict/batch")
def predict_dso3_batch(payload: DSO3BatchRequest) -> list[dict]:
    return _require_service().batch_predict_dso3(payload.rows)


@app.get("/dso1/evaluation", response_model=EvaluationResponse)
def dso1_evaluation() -> dict:
    return _require_service().dso1_evaluation()


@app.get("/dso2/evaluation", response_model=EvaluationResponse)
def dso2_evaluation() -> dict:
    return _require_service().dso2_evaluation()


@app.get("/dso3/overview", response_model=DSO3OverviewResponse)
def dso3_overview() -> dict:
    return _require_service().dso3_overview()


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "DSO Finance API is running",
        "docs": "/docs",
        "openapi": "/openapi.json",
    }
