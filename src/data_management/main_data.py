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
from src.data_management.ingestion.aemet_client import AemetClient  # ejemplo si tienes cliente de ingesta
from src.data_management.integration.regression.pipeline import run_pipeline

# Configuración logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Base del proyecto (subir un nivel desde src/)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

def run_ingestion():
    logger.info("Iniciando ingesta de datos desde APIs y archivos.")
    client = AemetClient()
    client.execute()   # tu lógica de ingesta
    logger.info("Ingesta completada correctamente.")

def run_integration():
    logger.info("Iniciando integración Claims + AEMET.")

    claims_path = BASE_DIR / "data" / "trusted" / "claims" / "weather_claims.parquet"
    aemet_path = BASE_DIR / "data" / "trusted" / "aemet_deltalake"
    cp_base_path = BASE_DIR / "data" / "external" / "data"
    output_path = BASE_DIR / "data" / "aggregated" / "claims_regression.parquet"

    run_pipeline(str(claims_path), str(aemet_path), str(cp_base_path), str(output_path))

    logger.info("Integración completada correctamente. Dataset disponible en aggregated.")

if __name__ == "__main__":
    # Paso 1. Ingesta
    #run_ingestion()

    # Paso 2. Integración
    run_integration()

    # (Opcional) Paso 3. Otros procesamientos downstream
    logger.info("Pipeline de datos completada ✅")
