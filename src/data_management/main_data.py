"""
Script principal para la ejecución de la pipeline de datos.

Responsabilidades:
- Ingesta de datos (APIs, ficheros externos)
- Procesamiento e integración (Claims + AEMET)
- Exportación a capa aggregated
"""

import logging
import os
from pathlib import Path
import pandas as pd

from src.data_management.ingestion.aemet_client import AemetClient  # ejemplo si tienes cliente de ingesta
from src.data_management.integration.regression.pipeline import run_pipeline as run_regression_pipeline
from src.data_management.integration.classification.pipeline import run_pipeline as run_classification_pipeline
import src.data_management.processing.aemet_trusted as aemet_processing_trusted

# Configuración logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Base del proyecto (subir un nivel desde src/)
BASE_DIR = Path(__file__).resolve().parent.parent.parent


# -------------------------------------------------
# Ingesta
# -------------------------------------------------
def run_ingestion():
    logger.info("Iniciando ingesta de datos desde APIs y archivos.")
    client = AemetClient()
    client.execute()   # tu lógica de ingesta
    logger.info("Ingesta completada correctamente.")


# -------------------------------------------------
# Integración: Modelo de Regresión
# -------------------------------------------------
def run_integration_regression():
    logger.info("Iniciando integración Claims + AEMET (regresión).")

    claims_path = BASE_DIR / "data" / "trusted" / "claims" / "weather_claims.parquet"
    aemet_path = BASE_DIR / "data" / "trusted" / "aemet_deltalake"
    cp_base_path = BASE_DIR / "data" / "external" / "data"
    output_path = BASE_DIR / "data" / "aggregated" / "claims_regression.parquet"

    run_regression_pipeline(str(claims_path), str(aemet_path), str(cp_base_path), str(output_path))

    logger.info("Integración de regresión completada ✅. Dataset disponible en aggregated.")


# -------------------------------------------------
# Integración: Modelo de Clasificación
# -------------------------------------------------
def run_integration_classification(start_year=2020, end_year=2025):
    logger.info(f"Iniciando integración Claims + AEMET (clasificación) de {start_year} a {end_year}.")

    claims_path = BASE_DIR / "data" / "trusted" / "claims" / "weather_claims.parquet"
    aemet_path = BASE_DIR / "data" / "trusted" / "aemet_deltalake"
    cp_base_path = BASE_DIR / "data" / "external" / "data"
    aggregated_dir = BASE_DIR / "data" / "aggregated"

    aggregated_dir.mkdir(parents=True, exist_ok=True)

    yearly_datasets = []
    for year in range(start_year, end_year + 1):
        output_path = aggregated_dir / f"claims_classification_{year}.parquet"

        logger.info(f"=== Ejecutando pipeline clasificación para {year} ===")
        run_classification_pipeline(
            claims_path=str(claims_path),
            aemet_path=str(aemet_path),
            cp_base_path=str(cp_base_path),
            output_path=str(output_path),
            year=year,
            k=4,                # hiperparámetros ajustados
            max_radius_km=40.0  # hiperparámetros ajustados
        )

        # Leer y acumular resultados
        yearly_datasets.append(pd.read_parquet(output_path))

    # Concatenar todos los años en un único dataset
    df_all = pd.concat(yearly_datasets, ignore_index=True)
    all_output_path = aggregated_dir / "claims_classification.parquet"
    df_all.to_parquet(all_output_path, index=False)

    logger.info(f"✅ Dataset de clasificación consolidado exportado a {all_output_path}, "
                f"con {df_all.shape[0]} registros y {df_all.shape[1]} columnas.")


# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":
    # Paso 0. Ingesta
    # run_ingestion()

    # Paso 1. Capa Trusted
    #aemet_processing_trusted.main()

    # Capa Aggregated
    # Paso 2. Integración regresión
    #run_integration_regression()

    # Paso 3. Integración clasificación
    run_integration_classification(2020, 2025)

    logger.info("Pipeline de datos completada ✅")
