import pandas as pd
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

    def _safe_get(self, url, max_retries=10, initial_wait=5):
        """
        Realiza una petición GET con reintentos y backoff exponencial.
        """
        for attempt in range(1, max_retries + 1):
            try:
                # Intentar hacer la petición GET con timeout de 30 segundos
                response = requests.get(url, headers=self.headers, timeout=30)

                # Si la respuesta es exitosa, devolverla inmediatamente
                if response.status_code == 200:
                    return response

                # Si se alcanza el límite de peticiones, aplicar backoff exponencial
                elif response.status_code == 429:
                    wait = initial_wait * attempt
                    logger.warning(f"Error 429 Too Many Requests. Reintentando en {wait}s... (Intento {attempt})")
                    time.sleep(wait)

                # Si hay errores temporales del servidor, también reintentar con backoff exponencial
                elif response.status_code in (500, 503):
                    wait = initial_wait * attempt
                    logger.warning(
                        f"Error {response.status_code} del servidor. Reintentando en {wait}s... (Intento {attempt})")
                    time.sleep(wait)

                # Otros errores HTTP (por ejemplo 403, 404) no se consideran recuperables
                else:
                    logger.error(f"Error HTTP {response.status_code} al acceder a {url}")
                    return None

            # Errores de conexión o timeout: posibles fallos temporales de red
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                # Tiempo de espera aumenta exponencialmente con cada intento
                wait = initial_wait * attempt
                logger.warning(f"Error de conexión: {e}. Reintentando en {wait}s... (Intento {attempt})")
                time.sleep(wait)

            # Si se agotan los intentos, registrar el fallo y devolver None
        logger.error(f"Fallo tras {max_retries} intentos para {url}")
        return None

    def get_stations(self):
        """
        Descarga el inventario completo de estaciones climatológicas
        y lo guarda como archivo Parquet.
        """
        try:
            logger.info("Requesting station inventory from AEMET...")
            resp = self._safe_get(self.url_stations, max_retries=5, initial_wait=5)
            if not resp:
                raise ValueError("Fallo en la descarga del inventario de estaciones")

            url_datos = resp.json().get("datos")
            if not url_datos:
                raise ValueError("Data URL not found in the initial response.")

            stations_resp = self._safe_get(url_datos)
            if not stations_resp:
                raise ValueError("Fallo en la descarga de los datos de estaciones")

            stations_data = stations_resp.json()

            # Guardar como Parquet (sin subcarpeta por año)
            filename = f"{self.name.lower()}_stations"
            self.save_parquet(
                filename=filename,
                content=stations_data,
                year=None # No particionar por año datos de estaciones
            )

            logger.info(f"Estaciones obtenidas: {len(stations_data)}")

        except Exception as e:
            logger.error(f"Error obteniendo estaciones: {e}", exc_info=True)

    def get_daily_climatology_all_stations(self, date: str):
        """
        Descarga las observaciones climatológicas diarias para TODAS las estaciones
        en una fecha específica y guarda el resultado como archivo Parquet.
        """
        base_url = "https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/datos"
        start_fmt = f"{date}T00:00:00UTC"
        end_fmt = f"{date}T23:59:59UTC"
        request_url = f"{base_url}/fechaini/{start_fmt}/fechafin/{end_fmt}/todasestaciones"

        logger.info(f"Solicitando valores climatológicos en fecha {date}")

        try:
            resp = self._safe_get(request_url)
            if not resp:
                raise ValueError("No se pudo obtener la primera respuesta de AEMET")

            first_resp = resp.json()
            datos_url = first_resp.get("datos")

            if not datos_url:
                raise ValueError("No se encontró URL de datos en la respuesta inicial")

            data_resp = self._safe_get(datos_url)
            if not data_resp:
                raise ValueError("No se pudo obtener los datos reales desde AEMET")

            climatology_data = data_resp.json()
            if not isinstance(climatology_data, list):
                raise ValueError("Respuesta inesperada: se esperaba una lista de observaciones")

            # Guardar como parquet en carpeta por año
            year = datetime.strptime(date, "%Y-%m-%d").year
            filename = f"{self.name.lower()}_datos_diarios_{date}"
            self.save_parquet(
                filename=filename,
                content=climatology_data,
                year=year
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
            result = self.get_daily_climatology_all_stations(date_str)

            if result is None:
                logger.warning(f"No se pudieron obtener datos para {date_str}")

            time.sleep(delay_seconds)
            current_date += timedelta(days=1)

    def execute_aemet(self):
        start_date = "2017-01-01"
        end_date = "2025-06-30"
        self.get_stations()
        self.get_daily_climatology_range(start_date, end_date, delay_seconds=10)