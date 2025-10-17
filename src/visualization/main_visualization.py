"""
Script para lanzar el dashboard (Streamlit u otra herramienta)
"""

import logging
from src.utils.logging_setup import setup_logging

setup_logging()
logger = logging.getLogger('main_visualization')

def run_visualization():
    logger.info("Iniciando módulo de Visualización")



if __name__ == "__main__":
    run_visualization()