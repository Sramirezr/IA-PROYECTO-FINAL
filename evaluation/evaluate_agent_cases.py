import csv
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agent.agent_chain import evaluate_candidate

CV_DIR = ROOT_DIR / "data" / "sample_cvs"
OUTPUT_PATH = ROOT_DIR / "evaluation" / "agent_results.csv"


JOB_DESCRIPTION = """
We are hiring a senior Python backend engineer with strong experience in FastAPI,
microservices, Docker, Kubernetes, AWS, CI/CD, PostgreSQL, Redis, and production
system monitoring. The role requires clear communication and ownership of
technical decisions.
"""


CASES = [
    ("cv_01_senior_dev.pdf", "alto"),
    ("cv_02_financial_analyst.pdf", "bajo"),
    ("cv_03_nurse.pdf", "bajo"),
    ("cv_04_marketing_manager.pdf", "bajo"),
    ("cv_05_civil_engineer.pdf", "bajo"),
    ("cv_06_junior_analyst.pdf", "medio"),
    ("cv_07_teacher.pdf", "bajo"),
    ("cv_08_accountant.pdf", "bajo"),
    ("cv_09_graphic_designer.pdf", "bajo"),
    ("cv_10_logistics.pdf", "bajo"),
    ("cv_11_chef.pdf", "bajo"),
    ("cv_12_history_grad.pdf", "bajo"),
    ("cv_13_retail_seller.pdf", "bajo"),
    ("cv_14_general_doctor.pdf", "bajo"),
    ("cv_15_freshman.pdf", "bajo"),
]


def score_to_level(score: int) -> str:
    if score >= 70:
        return "alto"
    if score >= 40:
        return "medio"
    return "bajo"


def main() -> None:
    load_dotenv()
    rows = []

    for index, (cv_filename, expected_level) in enumerate(CASES, start=1):
        cv_path = CV_DIR / cv_filename
        start = time.perf_counter()
        evaluation = evaluate_candidate(str(cv_path), JOB_DESCRIPTION)
        latency_sec = round(time.perf_counter() - start, 2)
        score = int(evaluation["score"])
        predicted_level = score_to_level(score)

        rows.append(
            {
                "cv_id": index,
                "expected_level": expected_level,
                "score": score,
                "predicted_level": predicted_level,
                "acerto": expected_level == predicted_level,
                "latency_sec": latency_sec,
                "fortalezas": evaluation.get("fortalezas", []),
                "brechas": evaluation.get("brechas", []),
                "recomendaciones": evaluation.get("recomendaciones", []),
            }
        )

        print(f"{index:02d} {cv_filename}: {score} ({predicted_level})")

    fieldnames = [
        "cv_id",
        "expected_level",
        "score",
        "predicted_level",
        "acerto",
        "latency_sec",
        "fortalezas",
        "brechas",
        "recomendaciones",
    ]
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    accuracy = sum(row["acerto"] for row in rows) / len(rows)
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Results written to {OUTPUT_PATH.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
