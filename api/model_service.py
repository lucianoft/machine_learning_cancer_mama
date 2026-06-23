from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PIPELINE_PATH = MODELS_DIR / "pipeline_breast_cancer.pkl"
METADATA_PATH = MODELS_DIR / "pipeline_metadata.pkl"


class ModelService:
    """Carrega e executa a pipeline treinada no notebook."""

    def __init__(self) -> None:
        self.pipeline: Pipeline | None = None
        self.metadata: dict | None = None

    def load(self) -> None:
        if not PIPELINE_PATH.exists():
            raise FileNotFoundError(f"Pipeline não encontrada: {PIPELINE_PATH}")
        if not METADATA_PATH.exists():
            raise FileNotFoundError(f"Metadados não encontrados: {METADATA_PATH}")

        self.pipeline = joblib.load(PIPELINE_PATH)
        self.metadata = joblib.load(METADATA_PATH)

    @property
    def feature_columns(self) -> list[str]:
        if not self.metadata:
            raise RuntimeError("Modelo não carregado.")
        return self.metadata["feature_columns"]

    def predict(self, patients: list[dict]) -> list[dict]:
        if not self.pipeline or not self.metadata:
            raise RuntimeError("Modelo não carregado.")

        df = pd.DataFrame(patients)
        missing = set(self.feature_columns) - set(df.columns)
        extra = set(df.columns) - set(self.feature_columns)
        if missing:
            raise ValueError(f"Colunas ausentes: {sorted(missing)}")
        if extra:
            raise ValueError(f"Colunas não esperadas: {sorted(extra)}")

        x = df[self.feature_columns]
        predictions = self.pipeline.predict(x)
        probabilities = self.pipeline.predict_proba(x)
        labels = self.metadata["target_labels"]

        results = []
        for i, pred in enumerate(predictions):
            pred_int = int(pred)
            results.append({
                "predicao": pred_int,
                "label": labels[pred_int],
                "probabilidade_benigno": float(probabilities[i][0]),
                "probabilidade_maligno": float(probabilities[i][1]),
            })
        return results


model_service = ModelService()
