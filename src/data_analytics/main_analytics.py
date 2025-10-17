import logging
from src.utils.logging_setup import setup_logging

import src.data_analytics.models.classification.logreg.train_logreg as c
import src.data_analytics.models.regression.modelo_regresion_GLM_GBT as r

setup_logging()
logger = logging.getLogger('main_analytics')

def run_analytics():

    logger.info(" Iniciando módulo de Data Analytics")

    # Modelo clasificacion
    logger.info("Ejecutando modelo de clasificación...")
    c.main()

    # Modelo regresion
    logger.info("Ejecutando modelo de regresión...")
    r.main()
    logger.info(" Data Analytics completado correctamente.")

if __name__ == "__main__":
    run_analytics()