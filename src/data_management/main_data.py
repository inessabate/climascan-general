"""
Script para la ingesta de datos:
- Llamadas a APIs
- Descarga de JSONs/Excels
- Guardado en data/landing y Delta Lake (data/aggregated)
"""

import logging
from src.utils.logging_setup import setup_logging
from src.data_management.ingestion.aemet_client import AemetClient

setup_logging()
logger = logging.getLogger(__name__)

def run_ingestion():

    logger.info("Iniciando  ingesta de datos desde APIs y archivos.")
    client = AemetClient()
    client.execute_aemet()
    logger.info("Ingesta completada correctamente.")


    logger.info("Iniciando procesamiento ETL con Spark.")
    #TODO: Llamar a la función de la etl entre capas aqui
    logger.info("Procesamiento completado correctamente.")

if __name__ == "__main__":
    run_ingestion()