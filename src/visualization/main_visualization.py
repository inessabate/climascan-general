"""
Script para lanzar el dashboard (Streamlit u otra herramienta)
"""

import logging
from src.utils.logging_setup import setup_logging
from src.visualization.plots import *

setup_logging()
logger = logging.getLogger('main_visualization')

def run_visualization():
    logger.info("Iniciando módulo de Visualización")

    generar_mapas_provincias(
        lista_provincias=["BARCELONA", "BADAJOZ", "ALAVA", "GIRONA"],
        ruta_csv="data/predictions/clasificacion_predictions_test_by_zip.csv",
        ruta_geojsons="data/external/data"
    )
    generar_mapas_coste(
        lista_provincias=["BARCELONA", "GIRONA", "CACERES", "CASTELLON"],
        ruta_csv="data/predictions/regresion_coste_siniestro.csv",
        ruta_geojsons="data/external/data",
    )


if __name__ == "__main__":
    run_visualization()