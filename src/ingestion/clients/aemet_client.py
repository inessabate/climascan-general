import requests
import os
import json
from dotenv import load_dotenv
from src.ingestion.clients.base_client import BaseClient

class AemetClient(BaseClient):
    def __init__(self):
        super().__init__("aemet")
        load_dotenv()
        self.api_key = os.getenv("API_KEY_AEMET")
        if not self.api_key:
            raise ValueError("API_KEY_AEMET not found in .env")
        self.headers = {
            "Accept": "application/json",
            "api_key": self.api_key
        }
        self.url_stations = "https://opendata.aemet.es/opendata/api/valores/climatologicos/inventarioestaciones/todasestaciones"

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

    def ejecutar(self):
        self.log(f"\nStarting {self.name.upper()} download...")
        self.get_stations()
        self.log(f"Finished data retrieval from {self.name.upper()}.")