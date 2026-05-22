from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.classifier.utils import clean_text


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "processed" / "jobs_clean.parquet"
MODELS_DIR = ROOT_DIR / "models"


def train_logreg_model():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "No se encontro data/processed/jobs_clean.parquet. Genera este archivo desde el notebook."
        )

    df = pd.read_parquet(DATA_PATH)
    x = df["text"].apply(clean_text)
    y = df["macro_category"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.9,
        stop_words="english",
        sublinear_tf=True,
    )
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    model = LogisticRegression(
        max_iter=200,
        class_weight="balanced",
        solver="liblinear",
        C=1.0,
    )
    model.fit(x_train_vec, y_train)
    predictions = model.predict(x_test_vec)

    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"F1-macro: {f1_score(y_test, predictions, average='macro'):.4f}")
    print(classification_report(y_test, predictions, digits=3))

    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(model, MODELS_DIR / "logreg_classifier.pkl")
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.pkl")


if __name__ == "__main__":
    train_logreg_model()
