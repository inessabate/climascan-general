"""
Climascan Data Pipeline - main.py

Este script orquesta todo el pipeline:
1️⃣ Data ingestion and management
2️⃣ Data analytics
4️⃣ Dashboard/visualization
"""


import src.data_analytics.main_analytics as a
import src.data_management.main_data as d
import src.visualization.main_visualization as v

import logging
from src.utils.logging_setup import setup_logging


setup_logging()
logger = logging.getLogger(__name__)

def main():
    logger.info("Running data ingestion...")
    d.run_ingestion()

    logger.info("Running data analytics...")
    a.run_analytics()

    logger.info("Running data visualization...")
    v.run_dashboard()


if __name__ == "__main__":

    main()