import logging
from src.utils.logging_setup import setup_logging

setup_logging()
logger = logging.getLogger('main_analytics')

def run_analytics():

    logger.info(" Iniciando módulo de Data Analytics")

    logger.info(" Data Analytics completado correctamente.")

if __name__ == "__main__":
    run_analytics()