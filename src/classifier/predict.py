from pathlib import Path

import joblib
import pandas as pd

from src.classifier.utils import clean_text


ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "models"


class MissingModelError(FileNotFoundError):
    pass


def _load_artifacts():
    vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    label_encoder_path = MODELS_DIR / "label_encoder.pkl"
    lgbm_path = MODELS_DIR / "lgbm_classifier.pkl"
    logreg_path = MODELS_DIR / "logreg_classifier.pkl"

    if not vectorizer_path.exists():
        raise MissingModelError(
            "No se encontro models/tfidf_vectorizer.pkl. Ejecuta el notebook de entrenamiento "
            "o copia los artefactos entrenados a la carpeta models/."
        )

    model_path = lgbm_path if lgbm_path.exists() else logreg_path
    if not model_path.exists():
        raise MissingModelError(
            "No se encontro models/lgbm_classifier.pkl ni models/logreg_classifier.pkl."
        )

    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path) if label_encoder_path.exists() else None
    return vectorizer, model, label_encoder, model_path.name


def predict_job_category(text: str) -> dict:
    vectorizer, model, label_encoder, model_name = _load_artifacts()

    cleaned = clean_text(text)
    if len(cleaned.split()) < 5:
        raise ValueError("La descripcion de la vacante es demasiado corta para clasificarla.")

    vectorized = vectorizer.transform([cleaned])

    if model_name.startswith("lgbm"):
        feature_names = vectorizer.get_feature_names_out()
        vectorized_for_model = pd.DataFrame.sparse.from_spmatrix(
            vectorized,
            columns=feature_names,
        )
    else:
        vectorized_for_model = vectorized

    predicted = model.predict(vectorized_for_model)[0]
    probabilities = model.predict_proba(vectorized_for_model)[0]

    if label_encoder is not None and not isinstance(predicted, str):
        predicted_label = label_encoder.inverse_transform([int(predicted)])[0]
        labels = list(label_encoder.classes_)
    else:
        predicted_label = str(predicted)
        labels = [str(label) for label in model.classes_]

    probability_by_class = {
        label: float(probability)
        for label, probability in zip(labels, probabilities)
    }

    top_probabilities = dict(
        sorted(probability_by_class.items(), key=lambda item: item[1], reverse=True)
    )

    return {
        "category": predicted_label,
        "probability": top_probabilities.get(predicted_label, 0.0),
        "probabilities": top_probabilities,
        "model": model_name,
    }
