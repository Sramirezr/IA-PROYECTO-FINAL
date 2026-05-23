# Clasificador de Ofertas de Empleo y Evaluador de CVs con IA

Sistema de inteligencia artificial que resuelve dos problemas del mercado laboral digital:
clasificar automaticamente ofertas de empleo por sector industrial, y evaluar la compatibilidad
de una hoja de vida frente a una vacante usando un agente de IA.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)
![LangGraph](https://img.shields.io/badge/Agent-LangGraph_ReAct-orange)
![Groq](https://img.shields.io/badge/LLM-Llama--3.1_via_Groq-purple)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

---

## Estructura del repositorio

```
IA-PROYECTO-FINAL/
|
|-- data/
|   |-- processed/          # jobs_clean.parquet (generado por el notebook)
|   |-- sample_cvs/         # 15 CVs sinteticos en PDF para evaluacion
|
|-- evaluation/
|   |-- agent_results.csv   # resultados de los 15 casos estandarizados
|
|-- models/
|   |-- tfidf_vectorizer.pkl
|   |-- lgbm_classifier.pkl
|   |-- logreg_classifier.pkl
|   |-- label_encoder.pkl
|
|-- notebooks/
|   |-- ProyectoFinal-IA.ipynb  # notebook principal
|
|-- reports/
|   |-- figures/            
|
|-- src/
|   |-- classifier/
|   |   |-- predict.py
|   |   |-- train.py
|   |-- agent/
|   |   |-- agent_chain.py
|   |   |-- cv_reader.py
|   |-- app/
|       |-- streamlit_app.py
|
|-- requirements.txt
|-- .env.example
```

---

## Instalacion y ejecucion

### 1. Clonar el repositorio

```bash
git clone https://github.com/Sramirezr/IA-PROYECTO-FINAL.git
cd IA-PROYECTO-FINAL
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Configurar variables de entorno

```bash
cp .env.example .env
# Editar .env y agregar GROQ_API_KEY
# Clave gratuita en: https://console.groq.com
```

### 4. Verificar artefactos

```bash
python scripts/check_project.py
```

Para validar tambien que existan los artefactos entrenados:

```bash
python scripts/check_project.py --strict-artifacts
```

### 5. Ejecutar la app

```bash
streamlit run src/app/streamlit_app.py
```

---

## Flujo recomendado

1. Ejecutar `notebooks/ProyectoFinal-IA.ipynb` para generar:
   - `data/processed/jobs_clean.parquet`
   - `models/tfidf_vectorizer.pkl`
   - `models/logreg_classifier.pkl` o `models/lgbm_classifier.pkl`
   - `models/label_encoder.pkl`

2. Si solo tienes `data/processed/jobs_clean.parquet`, puedes entrenar el baseline:

```bash
python src/classifier/train.py
```

3. Generar CVs sinteticos para la evaluacion:

```bash
python evaluation/generate_sample_cvs.py
```

4. Correr los 15 casos del agente:

```bash
python evaluation/evaluate_agent_cases.py
```

---

## Como usar la app

**Clasificador:** pega el texto de una vacante y el sistema predice su categoria industrial.

**Evaluador de CV:** sube un PDF de hoja de vida y pega la descripcion de la vacante.
El agente lee el PDF y devuelve un score de compatibilidad, fortalezas, brechas y recomendaciones.

---

## Dataset

LinkedIn Job Postings — Kaggle (`arshkon/linkedin-job-postings`). No esta incluido en el
repositorio. Se descarga automaticamente al ejecutar el notebook con `kagglehub`.

---

## Variables de entorno

| Variable | Descripcion | Requerida |
|---|---|---|
| `GROQ_API_KEY` | API de Groq para el agente LLM | Solo para Evaluador de CV |

El clasificador funciona sin API key. Solo el evaluador de CVs requiere Groq.


---

## Autores
Samuel Herrera Hoyos
Santiago Ramirez Ramirez
Mateo Villada Higuita
Proyecto final — Curso de Inteligencia Artificial
