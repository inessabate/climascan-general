"""
Script para lanzar el dashboard (Streamlit u otra herramienta)
"""

import logging
from src.utils.logging_setup import setup_logging

setup_logging()
logger = logging.getLogger('main_visualization')

def run_dashboard():
    logger.info("Iniciando módulo de Visualización")
    # TODO: Implementar lógica real (ejemplo subprocess)
    # import subprocess
    # subprocess.run(["streamlit", "run", "src/visualization/dashboard/app.py"])
    logger.info("Dashboard iniciado en http://localhost:8501")

if __name__ == "__main__":
    run_dashboard()