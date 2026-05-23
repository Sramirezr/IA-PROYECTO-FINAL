import json
import re

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.agent.cv_reader import extract_cv_text
from src.agent.prompts import EVALUATION_PROMPT


load_dotenv()


def _parse_json_response(raw_text: str) -> dict:
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError("El modelo no devolvio un JSON valido.")

    data = json.loads(match.group())
    data["score"] = max(0, min(100, int(data.get("score", 0))))

    for key in ["fortalezas", "brechas", "recomendaciones"]:
        if not isinstance(data.get(key), list):
            data[key] = []

    data.setdefault("resumen_perfil", "")
    return data


def evaluate_candidate(pdf_path: str, job_description: str, model: str = "llama-3.1-8b-instant") -> dict:
    if len(job_description.strip().split()) < 5:
        raise ValueError("La descripcion de la vacante es demasiado corta.")

    cv_text = extract_cv_text(pdf_path)
    prompt = ChatPromptTemplate.from_template(EVALUATION_PROMPT)
    llm = ChatGroq(model=model, temperature=0.2)
    chain = prompt | llm

    response = chain.invoke(
        {
            "job_description": job_description,
            "cv_text": cv_text[:12000],
        }
    )

    return _parse_json_response(response.content)
