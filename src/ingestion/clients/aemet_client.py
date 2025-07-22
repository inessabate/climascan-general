import requests
import os
import json
from dotenv import load_dotenv
from src.ingestion.clients.base_client import BaseClient
import src.ingestion.clients.urls as urls
from datetime import datetime


class AemetEndpoints:
    STATIONS = urls.AEMET_STATIONS
    STATION_DATA = urls.AEMET_STATION_DATA
    DAILY_DATA = urls.AEMET_DAILY_DATA
    MONTHLY_DATA = urls.AEMET_MONTHLY_DATA
    YEARLY_DATA = urls.AEMET_YEARLY_DATA
    DAILY_SUMMARY = urls.AEMET_DAILY_SUMMARY
    MONTHLY_SUMMARY = urls.AEMET_MONTHLY_SUMMARY
    YEARLY_SUMMARY = urls.AEMET_YEARLY_SUMMARY


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
        self.url_stations = AemetEndpoints.STATIONS

    def get_stations(self):
        try:
            self.log("Requesting station inventory from AEMET...")
            resp = requests.get(self.url_stations, headers=self.headers)
            resp.raise_for_status()
            url_datos = resp.json().get("datos")
            if not url_datos:
                raise ValueError("Data URL not found in the initial response.")

            stations_resp = requests.get(url_datos)
            stations_resp.raise_for_status()
            stations_data = stations_resp.json()

            self.save_json(f"{self.name.upper()}_stations", stations_data, include_date=False)
            self.log(f"Total stations downloaded: {len(stations_data)}")

        except Exception as e:
            self.log(f"Error getting stations: {e}")

    def get_daily_data(self, station_id: str, start_date: str, end_date: str):
        """Descarga datos diarios de una estación para un rango de fechas"""

        # Fechas en formato correcto para AEMET
        start_fmt = f"{start_date}T00:00:00UTC"
        end_fmt = f"{end_date}T23:59:59UTC"

        url_template = (
            f"{AemetEndpoints.DAILY_DATA}/fechaini/{start_fmt}/fechafin/{end_fmt}/estacion/{station_id}/"
        )

        print(f"[AEMET] Requesting daily data: {station_id} {start_date} -> {end_date}")

        try:
            resp = requests.get(url_template, headers=self.headers)
            resp.raise_for_status()
            data_url = resp.json().get("datos")

            if not data_url:
                raise ValueError(f"No hay datos disponibles para {station_id} ({start_date} - {end_date})")

            # Segundo request para obtener los datos reales
            daily_resp = requests.get(data_url)
            daily_resp.raise_for_status()
            daily_data = daily_resp.json()

            # Guardar
            filename = f"{self.name}_daily_{station_id}_{start_date}_{end_date}"
            self.save_json(filename, daily_data, include_date=False)
            self.log(f"Datos diarios guardados para {station_id} ({start_date} - {end_date})")

        except Exception as e:
            self.log(f"Error obteniendo datos diarios para {station_id}: {e}")

    def download_last_10_years(self):
        today = datetime.today()
        start_global = today.replace(year=today.year - 10)

        # Cargar estaciones desde el JSON previamente descargado
        stations_path = self.output_dir / "AEMET_stations.json"
        with open(stations_path, "r", encoding="utf-8") as f:
            stations = json.load(f)

        for station in stations:
            station_id = station.get("indicativo")
            if not station_id:
                continue

            print(f"\n=== Procesando estación {station_id} ===")

            # Iterar por años
            for year in range(start_global.year, today.year + 1):
                start_date = datetime(year, 1, 1)
                end_date = datetime(year, 12, 31)
                if end_date > today:
                    end_date = today

                start_str = start_date.strftime("%Y-%m-%d")
                end_str = end_date.strftime("%Y-%m-%d")

                self.get_daily_data(station_id, start_str, end_str)

    def execute_aemet(self):
        self.log(f"Starting {self.name.upper()} download...")
        self.get_stations()
        self.get_stations()
        # Luego descargar 10 años
        self.download_last_10_years()
        self.log(f"Finished data retrieval from {self.name.upper()}.")


