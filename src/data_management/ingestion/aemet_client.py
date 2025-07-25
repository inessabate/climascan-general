import requests
import os
import json
from dotenv import load_dotenv
from datetime import datetime, timedelta
import time
import logging
from src.utils.logging_setup import setup_logging
from src.data_management.ingestion.base_client import BaseClient


setup_logging()
logger = logging.getLogger()

class AemetClient(BaseClient):
    def __init__(self):
        super().__init__("aemet")
        load_dotenv()
        self.api_key = os.getenv("API_KEY_AEMET")
        if not self.api_key:
            raise RuntimeError("Falta la variable API_KEY_AEMET en .env")

        self.headers = {
            "Accept": "application/json",
            "api_key": self.api_key
        }
        self.url_stations = (
            "https://opendata.aemet.es/opendata/api/valores/climatologicos/inventarioestaciones/todasestaciones"
        )

    def get_stations(self):
        """Descarga el inventario completo de estaciones climatológicas"""
        try:
            logger.info(f"Requesting station inventory from AEMET...")
            resp = requests.get(self.url_stations, headers=self.headers)
            resp.raise_for_status()

            url_datos = resp.json().get("datos")
            if not url_datos:
                raise ValueError("Data URL not found in the initial response.")

            stations_resp = requests.get(url_datos)
            stations_resp.raise_for_status()
            stations_data = stations_resp.json()

            self.save_json(f"{self.name.upper()}_stations", stations_data, include_date=False)
            logger.info(f"Estaciones obtenidas: {len(stations_data)}")

        except Exception as e:
            logger.error(f"Error obteniendo estaciones: {e}", exc_info=True)

    def get_daily_climatology_all_stations(self, date: str):
        """
        Descarga las observaciones climatológicas diarias para TODAS las estaciones
        en una fecha específica.

        :param date: Fecha en formato YYYY-MM-DD
        """
        base_url = "https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/datos"
        start_fmt = f"{date}T00:00:00UTC"
        end_fmt = f"{date}T23:59:59UTC"
        request_url = f"{base_url}/fechaini/{start_fmt}/fechafin/{end_fmt}/todasestaciones"

        logger.info(f"Solicitando valores climatologicos en fecha {date}")

        try:
            # 1ª llamada -> devuelve URL con los datos reales
            resp = requests.get(request_url, headers=self.headers)
            resp.raise_for_status()
            first_resp = resp.json()
            datos_url = first_resp.get("datos")

            if not datos_url:
                raise ValueError("No se encontró URL de datos en la respuesta inicial")

            # 2ª llamada -> obtiene los datos reales
            data_resp = requests.get(datos_url)
            data_resp.raise_for_status()
            climatology_data = data_resp.json()

            self.save_json(
                f"{self.name.upper()}_datos_diarios_{date.replace('-', '')}",
                climatology_data,
                include_date=False
            )

            logger.info(
                f"Climatología diaria guardada para {date} "
                f"(Total registros: {len(climatology_data)})"
            )
            return climatology_data

        except Exception as e:
            logger.error(
                f"[{self.name}] Error obteniendo climatología diaria para {date}: {e}",
                exc_info=True,
            )
            return None


    def get_daily_climatology_range(self, start_date: str, end_date: str, delay_seconds: int = 5):
        """
        Descarga las observaciones climatológicas diarias para TODAS las estaciones
        en un rango de fechas.

        :param start_date: Fecha inicial en formato YYYY-MM-DD
        :param end_date: Fecha final en formato YYYY-MM-DD
        :param delay_seconds: Tiempo de espera entre llamadas (para evitar bloqueo)
        """
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")

        while current_date <= end_date_dt:
            date_str = current_date.strftime("%Y-%m-%d")

            # Descargar datos para esa fecha
            result = self.get_daily_climatology_all_stations(date_str)

            if result is None:
                logger.warning(f"No se pudieron obtener datos para {date_str}")
            else:
                pass

            # Esperar para no sobrecargar la API
            time.sleep(delay_seconds)

            # Avanzar un día
            current_date += timedelta(days=1)

    def execute_aemet(self):

        start_date = "2024-01-01"
        end_date = "2024-12-31"
        self.get_stations()
        self.get_daily_climatology_range(start_date, end_date, delay_seconds=10)

