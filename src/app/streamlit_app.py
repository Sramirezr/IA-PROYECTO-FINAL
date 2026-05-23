from pathlib import Path
import os
import sys
import tempfile

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

:root {
    --bg:        #0d0f14;
    --surface:   #161920;
    --surface2:  #1e2230;
    --border:    #2a2f42;
    --accent:    #4f8ef7;
    --accent2:   #a78bfa;
    --success:   #34d399;
    --warning:   #fbbf24;
    --danger:    #f87171;
    --text:      #e8eaf0;
    --muted:     #8b91a8;
    --radius:    14px;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: var(--bg) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

#MainMenu, footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

.hero-block {
    background: linear-gradient(135deg, #1a1f35 0%, #0f1520 60%, #12101e 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2.5rem 2.5rem 2rem;
    margin-bottom: 1.75rem;
    position: relative;
    overflow: hidden;
}
.hero-block::before {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(79,142,247,.18) 0%, transparent 70%);
}
.hero-block::after {
    content: "";
    position: absolute;
    bottom: -40px; left: 40px;
    width: 160px; height: 160px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(167,139,250,.12) 0%, transparent 70%);
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.1rem;
    letter-spacing: -0.03em;
    background: linear-gradient(90deg, #e8eaf0 30%, #4f8ef7 80%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 .35rem;
}
.hero-sub {
    color: var(--muted);
    font-size: .95rem;
    font-weight: 300;
    margin: 0;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: .4rem;
    background: rgba(79,142,247,.12);
    border: 1px solid rgba(79,142,247,.3);
    border-radius: 99px;
    padding: .25rem .75rem;
    font-size: .75rem;
    color: var(--accent);
    font-weight: 500;
    margin-top: .85rem;
}

.sidebar-logo {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.25rem;
    color: var(--text);
    margin-bottom: .1rem;
}
.sidebar-tag {
    font-size: .75rem;
    color: var(--muted);
    margin-bottom: 1.25rem;
}
.sidebar-pill {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: .55rem .85rem;
    font-size: .82rem;
    color: var(--muted);
    margin-bottom: .5rem;
    display: flex;
    align-items: center;
    gap: .5rem;
}
.sidebar-pill span { color: var(--text); font-weight: 500; }

[data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: var(--radius) !important;
    padding: .3rem !important;
    border: 1px solid var(--border) !important;
    gap: .2rem !important;
}

[data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 10px !important;
    color: var(--muted) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: .9rem !important;
    padding: .55rem 1.2rem !important;
    border: none !important;
    transition: all .2s ease !important;
}
[aria-selected="true"] {
    background: var(--surface2) !important;
    color: var(--text) !important;
    box-shadow: 0 1px 8px rgba(0,0,0,.35) !important;
}
[data-baseweb="tab-highlight"] { display: none !important; }
[data-baseweb="tab-border"] { display: none !important; }


[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--accent) 0%, #3a6fd8 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: .92rem !important;
    padding: .65rem 1.4rem !important;
    transition: all .2s ease !important;
    box-shadow: 0 4px 16px rgba(79,142,247,.25) !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(79,142,247,.38) !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

[data-baseweb="textarea"] textarea,
[data-baseweb="input"] input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: .92rem !important;
    transition: border-color .2s !important;
}
[data-baseweb="textarea"] textarea:focus,
[data-baseweb="input"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(79,142,247,.15) !important;
}
.stTextArea label, .stFileUploader label {
    color: var(--muted) !important;
    font-size: .82rem !important;
    font-weight: 500 !important;
    letter-spacing: .04em !important;
    text-transform: uppercase !important;
}

[data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1.1rem 1.3rem !important;
}
[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: .78rem !important;
    font-weight: 600 !important;
    letter-spacing: .08em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.65rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
}

[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
    border-radius: 99px !important;
}
[data-testid="stProgress"] > div {
    background: var(--surface2) !important;
    border-radius: 99px !important;
}

[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left-width: 3px !important;
}

hr { border-color: var(--border) !important; }

[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
}

[data-testid="stSpinner"] { color: var(--accent) !important; }

[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1rem !important;
    transition: border-color .2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}

.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: var(--text);
    margin: 1.25rem 0 .75rem;
    display: flex;
    align-items: center;
    gap: .5rem;
}
.section-header::after {
    content: "";
    flex: 1;
    height: 1px;
    background: var(--border);
}
</style>
""", unsafe_allow_html=True)


# ── Helper: probability chart ────────────────────────────────────────────────
def render_probability_chart(probabilities: dict):
    labels = list(probabilities.keys())[:8]
    values = list(probabilities.values())[:8]
    top_val = max(values) if values else 1

    colors = [
        f"rgba(79,142,247,{0.35 + 0.65 * (v / top_val):.2f})" for v in values
    ]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        text=[f"{v:.1%}" for v in values],
        textposition="outside",
        textfont=dict(color="#8b91a8", size=11, family="DM Sans"),
        marker=dict(
            color=colors,
            line=dict(color="rgba(79,142,247,.6)", width=1),
        ),
        hovertemplate="<b>%{y}</b><br>Probabilidad: %{x:.1%}<extra></extra>",
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#8b91a8"),
        xaxis=dict(
            tickformat=".0%",
            gridcolor="rgba(42,47,66,.6)",
            tickfont=dict(size=10),
            showline=False,
        ),
        yaxis=dict(
            categoryorder="total ascending",
            tickfont=dict(size=11, color="#e8eaf0"),
            showgrid=False,
        ),
        margin=dict(l=10, r=60, t=16, b=10),
        height=380,
        bargap=0.35,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Hero header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-block">
    <p class="hero-title">Clasificador de Ofertas de Empleo</p>
    <p class="hero-sub">Categorizacion inteligente de vacantes &middot; Evaluacion de compatibilidad CV</p>
    <div class="hero-badge">TF-IDF + Regresion Logistica &middot; Agente Groq</div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">Empleo IA</div>
    <div class="sidebar-tag">Clasificador inteligente de vacantes</div>
    <div class="sidebar-pill"><span>Modelo</span> TF-IDF + LogReg</div>
    <div class="sidebar-pill"><span>Agente CV</span> Groq LLM</div>
    <div class="sidebar-pill"><span>Notebook</span> ProyectoFinal_IA</div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:.75rem; color:#8b91a8; line-height:1.6;">
        Ingresa la descripcion de una vacante para clasificarla automaticamente,
        o sube un CV en PDF para evaluar su compatibilidad con una posicion.
    </div>
    """, unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_classifier, tab_cv = st.tabs([
    "Clasificador",
    "Evaluador de CV",
])


# ── TAB 1: Clasificador ───────────────────────────────────────────────────────
with tab_classifier:
    st.markdown('<div class="section-header">Descripcion de la vacante</div>', unsafe_allow_html=True)

    job_description = st.text_area(
        "Descripcion de la vacante",
        height=220,
        placeholder="Pega aqui el titulo, responsabilidades y requisitos de la vacante...",
        label_visibility="collapsed",
    )

    col_btn, col_hint = st.columns([1, 3])
    with col_btn:
        classify = st.button("Clasificar vacante", type="primary", use_container_width=True)
    with col_hint:
        st.markdown(
            '<p style="color:#8b91a8;font-size:.82rem;margin-top:.7rem;">El modelo analizara el texto y retornara la categoria mas probable junto con la distribucion de probabilidades.</p>',
            unsafe_allow_html=True,
        )

    if classify:
        if not job_description.strip():
            st.warning("Pega la descripcion de la vacante antes de clasificar.")
        else:
            try:
                result = predict_job_category(job_description)

                st.markdown('<div class="section-header">Resultado</div>', unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Categoria predicha", result["category"])
                with c2:
                    st.metric("Confianza", f"{result['probability']:.1%}")
                with c3:
                    st.metric("Modelo", result["model"])

                st.markdown('<div class="section-header">Distribucion de probabilidades</div>', unsafe_allow_html=True)
                render_probability_chart(result["probabilities"])

            except MissingModelError as error:
                st.error(str(error))
            except ValueError as error:
                st.warning(str(error))
            except Exception as error:
                st.error(f"No se pudo clasificar la vacante: {error}")


# ── TAB 2: Evaluador de CV ────────────────────────────────────────────────────
with tab_cv:
    col_upload, col_desc = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown('<div class="section-header">CV del candidato</div>', unsafe_allow_html=True)
        cv_file = st.file_uploader(
            "CV en PDF",
            type=["pdf"],
            label_visibility="collapsed",
        )
        if cv_file:
            st.markdown(f"""
            <div style="background:rgba(52,211,153,.08);border:1px solid rgba(52,211,153,.25);
                border-radius:10px;padding:.7rem 1rem;font-size:.85rem;color:#34d399;margin-top:.5rem;">
                {cv_file.name} cargado ({cv_file.size / 1024:.1f} KB)
            </div>
            """, unsafe_allow_html=True)

    with col_desc:
        st.markdown('<div class="section-header">Vacante objetivo</div>', unsafe_allow_html=True)
        cv_job_description = st.text_area(
            "Vacante para evaluar compatibilidad",
            height=160,
            placeholder="Pega aqui la vacante contra la que quieres evaluar el CV...",
            label_visibility="collapsed",
        )

    st.markdown("&nbsp;", unsafe_allow_html=True)
    evaluate_btn = st.button("Evaluar compatibilidad", type="primary", use_container_width=False)

    if evaluate_btn:
        if cv_file is None:
            st.warning("Sube un CV en PDF.")
        elif not cv_job_description.strip():
            st.warning("Pega la descripcion de la vacante.")
        elif not os.getenv("GROQ_API_KEY"):
            st.error("Falta configurar **GROQ_API_KEY** en el archivo `.env` o en el entorno.")
        else:
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(cv_file.getbuffer())
                    temp_path = tmp.name

                with st.spinner("Analizando compatibilidad con el agente IA..."):
                    evaluation = evaluate_candidate(temp_path, cv_job_description)

                score = evaluation["score"]
                score_color = "#34d399" if score >= 70 else "#fbbf24" if score >= 45 else "#f87171"

                st.markdown(f"""
                <div style="background:var(--surface);border:1px solid {score_color}33;
                    border-radius:var(--radius);padding:1.5rem;margin:1rem 0;text-align:center;">
                    <div style="font-size:.72rem;font-weight:600;letter-spacing:.1em;
                        text-transform:uppercase;color:#8b91a8;margin-bottom:.4rem;">
                        Score de compatibilidad
                    </div>
                    <div style="font-family:'Syne',sans-serif;font-size:3.2rem;font-weight:800;color:{score_color};line-height:1;">
                        {score}<span style="font-size:1.4rem;opacity:.6;">/100</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(score / 100)

                if evaluation.get("resumen_perfil"):
                    st.markdown(f"""
                    <div style="background:var(--surface2);border:1px solid var(--border);
                        border-radius:10px;padding:1rem 1.25rem;font-size:.9rem;color:#c8cad4;
                        line-height:1.7;margin:1rem 0;">
                        {evaluation['resumen_perfil']}
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown('<div class="section-header">Analisis detallado</div>', unsafe_allow_html=True)
                col_f, col_g = st.columns(2, gap="medium")

                with col_f:
                    st.markdown("**Fortalezas**")
                    for item in evaluation.get("fortalezas", []):
                        st.success(item)

                with col_g:
                    st.markdown("**Brechas**")
                    for item in evaluation.get("brechas", []):
                        st.warning(item)

                if evaluation.get("recomendaciones"):
                    st.markdown('<div class="section-header">Recomendaciones</div>', unsafe_allow_html=True)
                    for i, item in enumerate(evaluation["recomendaciones"], 1):
                        st.info(f"**{i}.** {item}")

            except CVReadError as error:
                st.error(f"Error leyendo el CV: {error}")
            except Exception as error:
                st.error(f"No se pudo evaluar el CV: {error}")
            finally:
                if temp_path and Path(temp_path).exists():
                    Path(temp_path).unlink()