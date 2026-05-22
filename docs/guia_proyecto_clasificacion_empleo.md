# Guia de Proyecto Final — Clasificacion de Ofertas de Empleo + Agente Evaluador de Hojas de Vida

## La Problematica

El mercado laboral digital genera miles de ofertas de empleo diarias en plataformas como LinkedIn, Indeed y Glassdoor. Dos problemas criticos emergen de esta abundancia:

**Del lado del reclutador:** Las vacantes publicadas son clasificadas manualmente o con sistemas rudimentarios de etiquetado. Esto genera inconsistencias en la indexacion, dificulta el analisis de demanda por sector, y hace imposible comparar tendencias entre industrias de forma automatizada.

**Del lado del candidato:** Evaluar si una hoja de vida es competitiva para una vacante especifica requiere tiempo, experiencia en el sector, y criterio. La mayoria de candidatos aplica sin saber realmente que tan compatible es su perfil, y recibe retroalimentacion —si la recibe— semanas despues.

**Lo que el sistema resuelve:** Un pipeline que clasifica automaticamente una oferta de empleo en su categoria industrial, y luego lanza un agente de IA que lee la hoja de vida del candidato, la compara contra el perfil del cargo clasificado, y entrega un score de compatibilidad con recomendaciones accionables en segundos.

---

## Stack Tecnologico Recomendado

| Componente | Herramienta | Razon |
|---|---|---|
| Clasificador de texto | TF-IDF + LightGBM (baseline) | Rapido, sin GPU, interpretable |
| Clasificador alternativo | DistilBERT via HuggingFace | Mejor accuracy si hay tiempo |
| Lectura de PDF | pdfplumber | Mas robusto que pypdf para CVs reales |
| Agente LLM | LangChain + Groq (Llama-3-70b) | Gratis, rapido, suficiente para este caso |
| Interfaz | Streamlit | Simple, visual, ideal para demo |
| Visualizaciones EDA | Plotly + Seaborn | Interactivo y estatico segun necesidad |
| Experimentos | MLflow (opcional) | Registro de metricas por run |
| Entorno | Google Colab + repositorio GitHub | Colab para entrenamiento, repo para entrega |

---

## Opciones de Dataset

### Opcion A — Job Postings Dataset (Kaggle) [RECOMENDADO]
- URL: `https://www.kaggle.com/datasets/arshkon/linkedin-job-postings`
- Tamano: ~120MB descomprimido
- Columnas utiles: `title`, `description`, `formatted_experience_level`, `skills_desc`
- Categorias disponibles: IT, Finance, Marketing, Healthcare, Engineering, Education, Legal, Sales
- Ventaja: limpio, balanceable, columnas ricas para EDA

### Opcion B — Indeed Job Postings Dataset
- URL: `https://www.kaggle.com/datasets/promptcloud/indeed-job-postings-dataset`
- Tamano: ~80MB
- Ventaja: mas variedad de sectores, descripciones mas largas
- Desventaja: requiere mas limpieza de texto

### Opcion C — Monster Job Listings
- URL: `https://www.kaggle.com/datasets/PromptCloudHQ/us-jobs-on-monstercom`
- Tamano: ~50MB
- Ventaja: el mas liviano, ideal si el tiempo es critico
- Desventaja: categorias menos definidas, requiere etiquetado manual de clases

### Opcion D — Synthetic + Real (Hibrido)
Combinar el dataset A con 200-300 vacantes exportadas manualmente de LinkedIn para categorias subrepresentadas. Solo si el desbalance de clases es severo.

---

## Estructura del Repositorio

```
proyecto_empleo_ia/
|
|-- data/
|   |-- raw/                    # Dataset original sin modificar
|   |-- processed/              # Dataset limpio y tokenizado
|   |-- sample_cvs/             # 15 CVs de prueba en PDF (para evaluacion del agente)
|
|-- notebooks/
|   |-- 01_eda.ipynb            # Analisis exploratorio completo
|   |-- 02_preprocessing.ipynb  # Limpieza y preparacion del texto
|   |-- 03_clasificador.ipynb   # Entrenamiento y evaluacion del modelo
|
|-- src/
|   |-- classifier/
|   |   |-- train.py            # Script de entrenamiento
|   |   |-- predict.py          # Inferencia sobre nuevas vacantes
|   |   |-- utils.py            # Limpieza de texto, tokenizacion
|   |
|   |-- agent/
|   |   |-- cv_reader.py        # Lectura y extraccion de texto desde PDF
|   |   |-- agent_chain.py      # Definicion del agente LangChain
|   |   |-- prompts.py          # Templates de prompts del agente
|   |
|   |-- app/
|       |-- streamlit_app.py    # Interfaz principal
|       |-- components.py       # Componentes reutilizables de UI
|
|-- evaluation/
|   |-- agent_test_cases.csv    # 15 casos estandarizados (CV + vacante + resultado esperado)
|   |-- agent_results.csv       # Resultados reales del agente sobre los 15 casos
|
|-- models/
|   |-- tfidf_vectorizer.pkl    # Vectorizador serializado
|   |-- lgbm_classifier.pkl     # Modelo serializado
|
|-- reports/
|   |-- informe_final.pdf       # Informe academico
|   |-- figures/                # Graficas exportadas del EDA y evaluacion
|
|-- requirements.txt
|-- README.md
|-- .env.example                # GROQ_API_KEY=tu_clave_aqui (nunca commitear .env real)
```

---

## Plan de Ejecucion — 3 Dias

### Dia 1 — Datos, EDA y Clasificador

#### Bloque 1: Configuracion del entorno (1-2h)
- [ ] Crear repositorio en GitHub con la estructura de carpetas anterior
- [ ] Crear entorno virtual o configurar Colab con `requirements.txt`
- [ ] Instalar dependencias: `lightgbm scikit-learn pdfplumber langchain langchain-groq streamlit plotly seaborn python-dotenv`
- [ ] Descargar dataset elegido desde Kaggle con la API: `kaggle datasets download -d arshkon/linkedin-job-postings`
- [ ] Verificar que el dataset carga correctamente con pandas y revisar columnas disponibles

#### Bloque 2: Analisis Exploratorio de Datos — EDA (2-3h)
- [ ] Cargar el dataset y hacer una inspeccion inicial: `.info()`, `.describe()`, `.isnull().sum()`
- [ ] Analizar la distribucion de clases (categorias de empleo) — graficar con Plotly bar chart
- [ ] Identificar desbalance de clases y decidir estrategia: submuestreo, sobremuestreo, o pesos en el modelo
- [ ] Analizar la longitud de las descripciones por categoria — histograma por clase
- [ ] Generar nube de palabras por categoria (WordCloud library)
- [ ] Analizar los terminos mas frecuentes por industria — grafica de barras horizontales por top-20 tokens
- [ ] Exportar todas las graficas a `reports/figures/`
- [ ] Documentar hallazgos en una celda Markdown del notebook

#### Bloque 3: Preprocesamiento de Texto (1-2h)
- [ ] Escribir funcion `clean_text(text)` que: convierta a minusculas, elimine HTML, URLs, caracteres especiales, y haga strip de espacios
- [ ] Eliminar stopwords en ingles con NLTK o spaCy
- [ ] Aplicar stemming o lematizacion (lematizacion recomendada con spaCy `en_core_web_sm`)
- [ ] Crear columna `text_clean` en el dataframe
- [ ] Codificar las etiquetas de categoria con `LabelEncoder` y guardar el encoder
- [ ] Hacer split estratificado: 80% train, 20% test — `train_test_split(..., stratify=y)`
- [ ] Guardar el dataset procesado en `data/processed/`

#### Bloque 4: Entrenamiento del Clasificador (2-3h)
- [ ] Entrenar vectorizador TF-IDF: `TfidfVectorizer(max_features=15000, ngram_range=(1,2))`
- [ ] Entrenar baseline con Logistic Regression para tener punto de comparacion
- [ ] Entrenar modelo principal LightGBM: `LGBMClassifier(class_weight='balanced', n_estimators=300)`
- [ ] Evaluar ambos modelos con: F1-macro, F1 por clase, Accuracy, Matriz de Confusion
- [ ] Graficar matriz de confusion con Seaborn heatmap
- [ ] Graficar F1 por clase con barras horizontales para identificar categorias problematicas
- [ ] Si hay tiempo: probar `DistilBERT` con HuggingFace Trainer como modelo alternativo
- [ ] Serializar el mejor modelo y el vectorizador con `joblib.dump()`
- [ ] Documentar las metricas finales en una tabla Markdown

---

### Dia 2 — Agente LangChain y Evaluacion

#### Bloque 5: Lector de CVs en PDF (1h)
- [ ] Crear `cv_reader.py` con funcion `extract_cv_text(pdf_path: str) -> str` usando `pdfplumber`
- [ ] Manejar el caso de CVs escaneados (texto vacio): lanzar excepcion clara con mensaje al usuario
- [ ] Extraer: nombre, experiencia, habilidades tecnicas, educacion — con un LLM o con regex simple
- [ ] Probar la extraccion sobre 3-5 CVs de muestra antes de conectar al agente

#### Bloque 6: Construccion del Agente LangChain (2-3h)
- [ ] Crear cuenta en Groq (groq.com) y obtener API key gratuita
- [ ] Configurar el LLM: `ChatGroq(model="llama3-70b-8192", temperature=0.3)`
- [ ] Definir el prompt del agente en `prompts.py`:
  - Contexto: descripcion de la vacante clasificada + categoria detectada
  - Tarea: leer el texto del CV, evaluar compatibilidad en escala 0-100, listar fortalezas, listar brechas, dar 3 recomendaciones concretas
  - Formato de salida: JSON estructurado con campos `score`, `fortalezas`, `brechas`, `recomendaciones`
- [ ] Crear la cadena en `agent_chain.py` usando `LLMChain` o `LCEL` (LangChain Expression Language)
- [ ] Definir la herramienta de lectura de PDF como `Tool` de LangChain: `Tool(name="read_cv", func=extract_cv_text, description="...")`
- [ ] Inicializar el agente: `initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)`
- [ ] Probar el agente manualmente con 2-3 casos antes de conectar a Streamlit

#### Bloque 7: Evaluacion Estandarizada del Agente (1-2h)
- [ ] Preparar 15 CVs de prueba en `data/sample_cvs/` — pueden ser CVs sinteticos generados con IA
- [ ] Para cada CV, definir: vacante de referencia, categoria esperada, score esperado (alto/medio/bajo)
- [ ] Correr el agente sobre los 15 casos y registrar resultados en `evaluation/agent_results.csv`
- [ ] Calcular tasa de acierto: comparar si la clasificacion de score (alto/medio/bajo) coincide con el esperado
- [ ] Documentar 3 casos de falla y analizar por que el agente erro

---

### Dia 3 — Interfaz Streamlit, Informe y Video

#### Bloque 8: Interfaz Streamlit (2-3h)
- [ ] Estructurar la app en pestanas con `st.tabs(["Clasificador de Vacantes", "Evaluador de CV", "EDA"])`

**Pestana 1 — Clasificador:**
- [ ] `st.text_area` para pegar descripcion de la vacante
- [ ] Boton "Clasificar" que llame a `predict.py` y muestre: categoria predicha, probabilidad por clase (grafica de barras Plotly), top-5 palabras clave que influyeron (si se usa modelo interpretable)

**Pestana 2 — Evaluador de CV:**
- [ ] `st.file_uploader` para subir el PDF del CV
- [ ] Campo para pegar o cargar la descripcion de la vacante
- [ ] Boton "Evaluar compatibilidad" que llame al agente y muestre:
  - Score de compatibilidad con `st.metric` y barra de progreso `st.progress`
  - Fortalezas en verde con `st.success`
  - Brechas en naranja con `st.warning`
  - Recomendaciones numeradas con `st.info`
- [ ] Spinner `st.spinner("El agente esta analizando tu perfil...")` durante la llamada al LLM

**Pestana 3 — EDA:**
- [ ] Mostrar las graficas del EDA embebidas con `st.plotly_chart`
- [ ] Metricas del modelo en `st.metric`: F1-macro, Accuracy, numero de clases

- [ ] Agregar sidebar con: descripcion del proyecto, stack utilizado, link al repositorio
- [ ] Probar la app completa localmente con `streamlit run src/app/streamlit_app.py`

#### Bloque 9: Informe Academico (2-3h)
Estructura sugerida (seguir la rubrica del curso):

1. Introduccion y planteamiento del problema
2. Descripcion del dataset y proceso de recoleccion
3. Analisis Exploratorio de Datos — incluir graficas exportadas
4. Arquitectura del sistema — diagrama de bloques del pipeline
5. Metodologia de entrenamiento — preprocesamiento, modelo, hiperparametros
6. Resultados del clasificador — tablas de metricas, matriz de confusion
7. Descripcion del agente — arquitectura LangChain, prompt engineering, herramientas
8. Evaluacion del agente — tabla con los 15 casos, tasa de acierto, analisis de fallos
9. Conclusiones y trabajo futuro
10. Referencias

- [ ] Escribir el informe en LaTeX o Google Docs segun lo que exija el curso
- [ ] Exportar todas las figuras del EDA e incluirlas en el informe
- [ ] Revisar que las metricas en el informe coincidan exactamente con las del notebook

#### Bloque 10: Video Demo (1-2h)
Estructura sugerida para el video (8-12 minutos):

- [ ] Introduccion: plantear el problema en 60 segundos con contexto real
- [ ] Demo en vivo de la app Streamlit:
  - Pegar una vacante real (ej: Senior Data Analyst en Bancolombia)
  - Mostrar la clasificacion en tiempo real con la grafica de probabilidades
  - Subir un CV de prueba y mostrar el output del agente (score, fortalezas, recomendaciones)
- [ ] Mostrar el notebook de EDA con las graficas mas llamativas
- [ ] Mostrar la matriz de confusion y explicar las categorias mas confundidas
- [ ] Cerrar con metricas del agente y conclusiones
- [ ] Grabar con OBS o Loom — resolucion minima 1080p

---

## Metricas de Evaluacion

### Clasificador
| Metrica | Descripcion | Umbral aceptable |
|---|---|---|
| F1-macro | Promedio de F1 por clase, sin ponderar por frecuencia | > 0.75 |
| F1 por clase | Para detectar clases problematicas | > 0.65 en todas |
| Accuracy | Solo referencial dado el posible desbalance | > 0.80 |
| Confusion Matrix | Analisis cualitativo de errores | Visualizacion obligatoria |

### Agente
| Metrica | Descripcion | Calculo |
|---|---|---|
| Tasa de acierto de clasificacion de score | Los 15 casos coinciden en nivel (alto/medio/bajo) | casos_acertados / 15 |
| Coherencia de recomendaciones | Evaluacion cualitativa: son especificas al cargo? | Rubrica 1-5 por caso |
| Latencia promedio | Tiempo de respuesta del agente | Medir con `time.time()` |

---

## Consideraciones Tecnicas Importantes

**Sobre el desbalance de clases:**
Es casi seguro que algunas categorias (IT, Sales) tendran muchas mas muestras que otras (Legal, Education). Usar siempre `class_weight='balanced'` en el modelo y `stratify=y` en el split. Reportar F1-macro, no accuracy.

**Sobre el prompt del agente:**
El prompt es el componente mas critico del agente. Ser muy especifico sobre el formato de salida (JSON) y pedir que el score este justificado con evidencia del CV. Probar al menos 3 versiones del prompt antes de usar la definitiva.

**Sobre la seguridad de la API key:**
Nunca hardcodear `GROQ_API_KEY` en el codigo. Usar `python-dotenv` con un archivo `.env` que este en `.gitignore`. En Colab usar `from google.colab import userdata`.

**Sobre el manejo de CVs en PDF:**
`pdfplumber` funciona bien con CVs digitales. CVs escaneados o con mucho formato grafico van a retornar texto vacio o roto. Agregar validacion: si el texto extraido es menor a 100 caracteres, mostrar mensaje de error al usuario.

**Sobre Streamlit en produccion:**
Para el demo del video, correr la app localmente. Si se quiere deploy publico gratuito para la entrega, usar Streamlit Community Cloud (streamlit.io/cloud) — se conecta directamente al repositorio de GitHub.

---

## Checklist Final de Entrega

- [ ] Repositorio GitHub publico con README claro y estructura de carpetas completa
- [ ] Notebook de EDA con minimo 6 visualizaciones documentadas
- [ ] Notebook de entrenamiento con metricas finales tabuladas
- [ ] Modelo serializado y reproducible (seed fija, requirements.txt con versiones)
- [ ] Agente funcional con logs de los 15 casos de prueba
- [ ] App Streamlit funcionando sin errores en las 3 pestanas
- [ ] Informe con todas las secciones de la rubrica cubiertas
- [ ] Video demo de 8-12 minutos con demo en vivo
- [ ] `.env.example` en el repo (nunca el `.env` real)
- [ ] `requirements.txt` con versiones exactas (`pip freeze > requirements.txt`)
