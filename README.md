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
│   ├── trusted/               # Datos limpios y validados
│   │   ├── claims/            # Datos de siniestros en formato tabular con data quality aplicado
│   │   └── meteo/             # Datos meteorológicos en formato tabular con data quality aplicado
│   │ 
│   └── aggregated/            # Datos agregados en Delta Lake, listos para data analytics
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
```
---

## 📊 Flujo de Datos – Proyecto Climascan

El pipeline de datos sigue un enfoque **medallion architecture** (Bronze → Silver → Gold), adaptado a nuestro proyecto:


### 🔹 Landing
- **raw_aemet/** → datos brutos descargados desde la API AEMET.  
- **raw_claims/** → datos iniciales de siniestros.

### 🔹 Trusted
- **trusted/aemet_deltalake/** → datos de AEMET limpios, transformados y normalizados.  
- **trusted/claims/weather_claims.parquet** → siniestros estandarizados y enriquecidos con centroides de CP.

### 🔹 Aggregated
- **aggregated/claims_enriched.parquet** → dataset final enriquecido con interpolación k-NN de variables meteorológicas.


### 🔹 Data Sources External
- **GeoJSON Códigos Postales** → centroides de códigos postales para georreferenciación.

Los archivos de códigos postales por provincia utilizados en este proyecto se obtienen desde el repositorio público de [Íñigo Flores](https://github.com/inigoflores/ds-codigos-postales).
Para reproducir la descarga de los archivos `.geojson`, puede ejecutarse el siguiente script en Python:

```python
import os
import requests

provincias = [
    "A_CORUNA", "ALAVA", "ALBACETE", "ALICANTE", "ALMERIA", "ASTURIAS", "AVILA", "BADAJOZ",
    "BALEARES", "BARCELONA", "BURGOS", "CACERES", "CADIZ", "CANTABRIA", "CASTELLON",
    "CEUTA", "CIUDAD_REAL", "CORDOBA", "CUENCA", "GIRONA", "GRANADA", "GUADALAJARA",
    "GIPUZCOA", "HUELVA", "HUESCA", "JAEN", "LA_RIOJA", "LAS_PALMAS",
    "LEON", "LLEIDA", "LUGO", "MADRID", "MALAGA", "MELILLA", "MURCIA", "NAVARRA",
    "OURENSE", "PALENCIA", "PONTEVEDRA", "SALAMANCA",
    "SEGOVIA", "SEVILLA", "SORIA", "TARRAGONA", "TERUEL", "TENERIFE",
    "TOLEDO", "VALENCIA", "VALLADOLID", "VIZCAYA", "ZAMORA", "ZARAGOZA"
]

BASE_URL = "https://raw.githubusercontent.com/inigoflores/ds-codigos-postales/master/data/"
DEST_DIR = "data/external/data"
os.makedirs(DEST_DIR, exist_ok=True)

for prov in provincias:
    file_url = f"{BASE_URL}{prov}.geojson"
    dest_path = os.path.join(DEST_DIR, f"{prov}.geojson")
    try:
        print(f"Descargando {prov}...", end=" ")
        r = requests.get(file_url, timeout=30)
        if r.status_code == 200:
            with open(dest_path, "wb") as f:
                f.write(r.content)
            print("OK")
        else:
            print(f"Error HTTP {r.status_code}")
    except Exception as e:
        print(f"Error con {prov}: {e}")

print("\n Descarga completada.")
```

Los archivos obtenidos deben ubicarse dentro del repositorio, en ```
data/external/data```.

---

### 🔄 Flujo resumido
1. **Ingesta** → descarga de datos desde APIs y archivos externos.  
2. **Landing** → almacenamiento en bruto (raw).  
3. **Trusted** → limpieza, normalización y unión con centroides CP.  
4. **Aggregated** → interpolación espacial (k-NN) y enriquecimiento final.  
