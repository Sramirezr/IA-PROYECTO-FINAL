from pathlib import Path
import argparse
import os
import sys

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


REQUIRED_PATHS = [
    "notebooks/ProyectoFinal_IA.ipynb",
    "requirements.txt",
    "streamlit_app.py",
    "src/app/streamlit_app.py",
    "src/classifier/predict.py",
    "src/classifier/train.py",
    "src/agent/agent_chain.py",
    "src/agent/cv_reader.py",
    "src/agent/prompts.py",
    "evaluation/generate_sample_cvs.py",
    "evaluation/evaluate_agent_cases.py",
    "docs/guia_proyecto_clasificacion_empleo.md",
    "data/sample_cvs",
]


ARTIFACT_PATHS = [
    "data/processed/jobs_clean.parquet",
    "models/tfidf_vectorizer.pkl",
    "models/logreg_classifier.pkl",
    "models/label_encoder.pkl",
]


OPTIONAL_PATHS = [
    "models/lgbm_classifier.pkl",
    "reports/figures/03_confusion_matrix_logreg.png",
    "evaluation/agent_results.csv",
]


def print_status(label: str, path_text: str, required: bool = True) -> bool:
    path = ROOT_DIR / path_text
    exists = path.exists()
    status = "OK" if exists else ("FALTA" if required else "opcional")
    print(f"[{status:7}] {path_text}")
    return exists


def main() -> int:
    parser = argparse.ArgumentParser(description="Revisa la estructura del proyecto.")
    parser.add_argument(
        "--strict-artifacts",
        action="store_true",
        help="Falla si faltan datos procesados o modelos entrenados.",
    )
    args = parser.parse_args()

    print("Revision del proyecto\n")

    required_ok = True
    for path_text in REQUIRED_PATHS:
        required_ok = print_status("base", path_text) and required_ok

    print("\nArtefactos generados")
    artifacts_ok = True
    for path_text in ARTIFACT_PATHS:
        artifacts_ok = print_status("artifact", path_text) and artifacts_ok

    print("\nOpcionales")
    for path_text in OPTIONAL_PATHS:
        print_status("optional", path_text, required=False)

    print("\nEntorno")
    groq_status = "OK" if os.getenv("GROQ_API_KEY") else "FALTA"
    print(f"[{groq_status:7}] GROQ_API_KEY")

    print("\nResumen")
    if required_ok and artifacts_ok:
        print("Proyecto listo para ejecutar clasificador y app.")
        return 0

    if required_ok and not artifacts_ok:
        print("La base del proyecto esta lista, pero faltan artefactos del notebook o del ZIP.")
        return 1 if args.strict_artifacts else 0

    print("Faltan archivos base del proyecto.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
