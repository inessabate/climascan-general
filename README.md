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
climascan-data-pipeline/
│
├── README.md                  # Documentación del proyecto
├── main.py                    # Orquestador principal del pipeline
│
├── config/                    # Archivos de configuración global
│   ├── config.yaml            # Parámetros del pipeline (rutas, fechas, etc.)
│   └── logging.yaml           # Configuración del sistema de logging
│
├── data/                      # Almacenamiento de datos en Delta Lake
│   ├── landing/               # Datos crudos (respuestas JSON). No se incluyen datos de siniestros por razones de privacidad.
│   │   └── aemet/             # Datos crudos de la API de AEMET (JSON). Una carpeta por año.
│   │  
│   ├── trusted/               # Datos validados / curados listos para procesar
│   │   ├── claims/            # Datos de siniestros en formato tabular con data quality aplicado
│   │   └── meteo/             # Datos meteorológicos en formato tabular con data quality aplicado
│   │ 
│   └── aggregated/            # Datos agregados en Delta Lake
│       ├── claims/            # Datos de siniestros en Delta Lake
│       └── meteo/             # Datos meteorológicos en Delta Lake
│
│
├── notebooks/                 # Exploración y análisis 
│   ├── 01_data_analytics/     # Análisis, consultas y features
│   └── 02_visualization/      # Prototipos de visualización y dashboards
│
├── requirements.txt           # Dependencias del entorno Python
│
├── scripts/                   # Scripts ejecutables para lanzar cada módulo
│   ├── run_ingestion.py       # Lanza el proceso de ingestión de datos
│   ├── run_processing.py      # Ejecuta las transformaciones/ETL
│   ├── run_analytics.py       # Lanza consultas y modelos de analítica
│   └── run_dashboard.py       # Despliega el dashboard de visualización
│
├── src/                       # Código fuente del proyecto
│   ├── data_management/       # Módulo 1️⃣: Ingestión y procesamiento de datos
│   │   ├── ingestion/         # Llamadas a APIs y carga de datos
│   │   ├── processing/        # Limpieza y ETL con Spark u otras herramientas
│   │   ├── utils/             # Funciones auxiliares para ingestión
│   │   └── main_data.py       # Punto de entrada del módulo de data management
│   │
│   ├── data_analytics/        # Módulo 2️⃣: Consultas y modelos de analítica
│   │   ├── querying/          # Consultas SQL/DuckDB sobre Delta Lake
│   │   ├── models/            # Modelos ML/NN para analítica avanzada
│   │   ├── utils/             # Funciones auxiliares para analítica
│   │   └── main_analytics.py  # Punto de entrada del módulo de analytics
│   │
│   ├── visualization/         # Módulo 3️⃣: Visualización y dashboards
│   │   ├── dashboard/         # Código de Streamlit u otras herramientas
│   │   ├── plots/             # Scripts para generar gráficos estáticos
│   │   ├── utils/             # Funciones auxiliares para visualización
│   │   └── main_visualization.py # Punto de entrada del módulo de visualización
│   │
│   └── utils/                 # Utilidades globales para todo el proyecto
│       └── logging_setup.py   # Configuración centralizada de logs
│
└── tests/                     # Pruebas unitarias por módulo
├── test_data_management/
├── test_data_analytics/
└── test_visualization/
```
