from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.model_service import model_service
from api.schemas import (
    HealthResponse,
    MetadataResponse,
    PredictRequest,
    PredictResponse,
)

API_DESCRIPTION = """
API de apoio à classificação **benigno vs maligno** em massas mamárias, com base em
características de núcleos celulares extraídas de imagens de **PAAF** (Punção Aspirativa
por Agulha Fina).

**Dataset:** Breast Cancer Wisconsin (Diagnostic) — UCI / Kaggle.

> Ferramenta de **apoio à decisão**. Não substitui avaliação médica, mamografia,
> ultrassonografia nem biópsia.
"""


@asynccontextmanager
async def lifespan(_app: FastAPI):
    model_service.load()
    yield


app = FastAPI(
    title="Tech Challenge — Câncer de Mama (Breast Cancer Wisconsin)",
    description=API_DESCRIPTION,
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
def health() -> HealthResponse:
    """Verifica se a API e o modelo estão carregados."""
    return HealthResponse(
        status="ok",
        model_name=model_service.metadata.get("model_name") if model_service.metadata else None,
    )


@app.get("/metadata", response_model=MetadataResponse, tags=["Sistema"])
def metadata() -> MetadataResponse:
    """Retorna metadados do modelo treinado para câncer de mama."""
    meta = model_service.metadata or {}
    return MetadataResponse(
        model_name=meta["model_name"],
        target_column=meta["target_column"],
        target_labels=meta["target_labels"],
        feature_columns=meta["feature_columns"],
        dataset=meta["dataset"],
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Predição — câncer de mama"],
    summary="Classificação benigno/maligno",
)
def predict(request: PredictRequest) -> PredictResponse:
    """
    Estima se a amostra é **benigna** ou **maligna** com base nas 30 medidas
    de núcleos celulares (mean, error e worst).
    """
    try:
        results = model_service.predict([request.patient.to_feature_dict()])
        return PredictResponse(**results[0])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
