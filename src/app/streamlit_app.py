from pathlib import Path
import os
import sys
import tempfile

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.classifier.predict import MissingModelError, predict_job_category
from src.agent.agent_chain import evaluate_candidate
from src.agent.cv_reader import CVReadError


st.set_page_config(
    page_title="Clasificador de Empleo IA",
    layout="wide",
)


def render_probability_chart(probabilities: dict):
    df = pd.DataFrame(
        {
            "Categoria": list(probabilities.keys())[:8],
            "Probabilidad": list(probabilities.values())[:8],
        }
    )
    fig = px.bar(
        df,
        x="Probabilidad",
        y="Categoria",
        orientation="h",
        text=df["Probabilidad"].map(lambda value: f"{value:.1%}"),
        color="Categoria",
    )
    fig.update_layout(
        showlegend=False,
        yaxis={"categoryorder": "total ascending"},
        margin=dict(l=10, r=10, t=20, b=10),
        height=420,
    )
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)


st.title("Clasificador de Ofertas de Empleo")

with st.sidebar:
    st.header("Proyecto")
    st.write("TF-IDF + modelo supervisado")
    st.write("Evaluador de CVs con Groq")
    st.write("Notebook base: `ProyectoFinal_IA.ipynb`")

tab_classifier, tab_cv, tab_status = st.tabs(
    ["Clasificador", "Evaluador de CV", "Estado"]
)

with tab_classifier:
    job_description = st.text_area(
        "Descripcion de la vacante",
        height=260,
        placeholder="Pega aqui el titulo, responsabilidades y requisitos de la vacante...",
    )

    col_action, col_result = st.columns([1, 2])
    with col_action:
        classify = st.button("Clasificar", type="primary", use_container_width=True)

    if classify:
        try:
            result = predict_job_category(job_description)
            with col_result:
                st.metric("Categoria predicha", result["category"])
                st.metric("Confianza", f"{result['probability']:.1%}")
                st.caption(f"Modelo usado: {result['model']}")

            render_probability_chart(result["probabilities"])
        except MissingModelError as error:
            st.error(str(error))
        except ValueError as error:
            st.warning(str(error))
        except Exception as error:
            st.error(f"No se pudo clasificar la vacante: {error}")

with tab_cv:
    cv_file = st.file_uploader("CV en PDF", type=["pdf"])
    cv_job_description = st.text_area(
        "Vacante para evaluar compatibilidad",
        height=220,
        placeholder="Pega aqui la vacante contra la que quieres evaluar el CV...",
    )

    if st.button("Evaluar compatibilidad", type="primary", use_container_width=True):
        if cv_file is None:
            st.warning("Sube un CV en PDF.")
        elif not cv_job_description.strip():
            st.warning("Pega la descripcion de la vacante.")
        elif not os.getenv("GROQ_API_KEY"):
            st.error("Falta configurar GROQ_API_KEY en el archivo .env o en el entorno.")
        else:
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(cv_file.getbuffer())
                    temp_path = temp_file.name

                with st.spinner("Analizando compatibilidad..."):
                    evaluation = evaluate_candidate(temp_path, cv_job_description)

                st.metric("Score de compatibilidad", f"{evaluation['score']}/100")
                st.progress(evaluation["score"] / 100)
                st.write(evaluation.get("resumen_perfil", ""))

                col_strengths, col_gaps = st.columns(2)
                with col_strengths:
                    st.subheader("Fortalezas")
                    for item in evaluation.get("fortalezas", []):
                        st.success(item)

                with col_gaps:
                    st.subheader("Brechas")
                    for item in evaluation.get("brechas", []):
                        st.warning(item)

                st.subheader("Recomendaciones")
                for index, item in enumerate(evaluation.get("recomendaciones", []), start=1):
                    st.info(f"{index}. {item}")
            except CVReadError as error:
                st.error(str(error))
            except Exception as error:
                st.error(f"No se pudo evaluar el CV: {error}")
            finally:
                if temp_path and Path(temp_path).exists():
                    Path(temp_path).unlink()

with tab_status:
    artifacts = {
        "data/processed/jobs_clean.parquet": ROOT_DIR / "data" / "processed" / "jobs_clean.parquet",
        "models/tfidf_vectorizer.pkl": ROOT_DIR / "models" / "tfidf_vectorizer.pkl",
        "models/logreg_classifier.pkl": ROOT_DIR / "models" / "logreg_classifier.pkl",
        "models/label_encoder.pkl": ROOT_DIR / "models" / "label_encoder.pkl",
        "evaluation/agent_results.csv": ROOT_DIR / "evaluation" / "agent_results.csv",
    }
    rows = [
        ("Estructura del proyecto", "Listo"),
        ("Codigo de prediccion", "Listo"),
        ("App Streamlit", "Listo"),
        ("Agente de CV", "Listo"),
        (
            "Dataset procesado",
            "Listo" if artifacts["data/processed/jobs_clean.parquet"].exists() else "Pendiente",
        ),
        (
            "Modelos entrenados",
            "Listo"
            if all(
                artifacts[path].exists()
                for path in [
                    "models/tfidf_vectorizer.pkl",
                    "models/logreg_classifier.pkl",
                    "models/label_encoder.pkl",
                ]
            )
            else "Pendiente",
        ),
        (
            "Evaluacion de 15 casos",
            "Listo" if artifacts["evaluation/agent_results.csv"].exists() else "Pendiente",
        ),
    ]
    st.dataframe(
        pd.DataFrame(rows, columns=["Componente", "Estado"]),
        use_container_width=True,
        hide_index=True,
    )
