# Climascan Data Pipeline

Este repositorio implementa un pipeline de datos diseñado para construir un modelo de predicción de riesgo ante eventos climáticos extremos. El proyecto sigue una estructura basada en el estándar [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/), adaptada a un flujo de trabajo con Delta Lake, datos meteorológicos y datos de siniestros.

---
## 🎯 Objetivo del Repositorio

Este repositorio tiene como objetivo **recoger, procesar y almacenar datos** relacionados con eventos climáticos extremos y siniestros asociados, con el fin de construir un conjunto de datos limpio y estructurado que sirva como base para el entrenamiento de modelos de predicción de riesgo.

El flujo general del pipeline incluye:

- 📥 **Ingesta** de datos meteorológicos desde APIs y datos de siniestros desde ficheros Excel.
- 🧹 **Procesamiento y transformación** de los datos para limpieza, enriquecimiento y creación de variables.
- 💾 **Almacenamiento** en formato Delta Lake para consultas eficientes y uso posterior en modelado predictivo.
---


## 📁 Estructura del Proyecto

```plaintext
Climascan-data-pipeline/
│
├── data/                      # Almacenamiento local de datos
│   ├── raw/                   # Datos sin procesar (por ejemplo, Excels originales o respuestas JSON)
│   ├── interim/               # Datos parcialmente procesados o fusionados
│   ├── processed/             # Datos listos para entrenar modelos (features)
│   └── delta/                 # Tablas en formato Delta Lake
│       ├── claims/            # Datos de siniestros transformados en Delta
│       └── meteo/             # Datos meteorológicos transformados en Delta
│
├── notebooks/                 # Jupyter/Databricks notebooks de exploración y EDA
│
├── src/                       # Código fuente del pipeline
│   ├── ingestion/             # Scripts para cargar datos desde APIs y Excels
│   ├── processing/            # Transformaciones, limpiezas y feature engineering
│   └── utils/                 # Funciones auxiliares (creación de sesión Spark, logs, helpers)
│
├── config/
│   └── config.yaml            # Parámetros globales del pipeline (rutas, fechas, etc.)
│
├── tests/                     # Pruebas unitarias del pipeline
│
├── requirements.txt           # Dependencias del entorno
├── .env                       # Variables de entorno (no versionadas)
├── .gitignore                 # Exclusión de carpetas como `.idea/`, `venv/`, `.env`, etc.
└── README.md                  # Este archivo

```
