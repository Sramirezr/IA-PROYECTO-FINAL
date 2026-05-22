# Clasificador de Ofertas de Empleo y Evaluador de CVs

Proyecto de IA para clasificar ofertas de empleo por macro-categoria y evaluar la compatibilidad de una hoja de vida frente a una vacante.

## Estado rapido

Revisar primero:

```bash
python scripts/check_project.py
```

Para validar tambien que existan los artefactos entrenados:

```bash
python scripts/check_project.py --strict-artifacts
```

El proyecto incluye los modelos entrenados necesarios para que el clasificador funcione:

- `models/tfidf_vectorizer.pkl`
- `models/logreg_classifier.pkl`
- `models/label_encoder.pkl`

Para reentrenar desde cero tambien se necesita `data/processed/jobs_clean.parquet`.
Si ese archivo falta, ejecuta el notebook `notebooks/ProyectoFinal_IA.ipynb` y
descarga/extrae el ZIP de artefactos en la raiz del proyecto.

## Componentes

- Clasificador de texto con TF-IDF y modelos supervisados.
- Evaluador de CVs en PDF usando `pdfplumber` y Groq.
- Interfaz web con Streamlit.
- Evaluacion estandarizada con 15 casos de prueba.

## Estructura

```text
data/
  raw/
  processed/
  sample_cvs/
docs/
notebooks/
models/
reports/
  figures/
evaluation/
scripts/
src/
  classifier/
  agent/
  app/
```

## Ejecucion

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar la app:

```bash
streamlit run src/app/streamlit_app.py
```

Tambien se puede ejecutar desde el archivo raiz:

```bash
streamlit run streamlit_app.py
```

## Flujo recomendado

1. Ejecutar `notebooks/ProyectoFinal_IA.ipynb` para generar:
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

5. Ejecutar la app:

```bash
streamlit run streamlit_app.py
```

## Variables de entorno

Crear un archivo `.env` a partir de `.env.example` y configurar:

```text
GROQ_API_KEY=...
```

## Subida a GitHub

Ver `docs/SUBIR_A_GITHUB.md`.
