from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import logging
from src.utils.logging_setup import setup_logging

setup_logging()
logger = logging.getLogger()

class BaseClient:
    def __init__(self, source_name: str):
        self.name = source_name.upper()
        self.base_output_dir = (
                Path(__file__).resolve().parent.parent.parent.parent
                / "data"
                / "landing"
                / source_name.lower()
        )
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.today_str = datetime.today().strftime("%Y-%m-%d")

    def save_json(self, filename: str, content: dict | list, include_date: bool = True, year: int = None):
        """
        Guarda contenido JSON en un archivo.

        Parámetros:
        ----------
        filename : str
            Nombre base del archivo (sin extensión .json).
        content : dict | list
            Contenido a guardar en formato JSON.
        include_date : bool
            Si True, añade la fecha actual al nombre del archivo.
        year : int
            Si se especifica, guarda el archivo dentro de una subcarpeta con ese año.
        """
        # Construcción del nombre de archivo
        if include_date:
            full_filename = f"{filename}_{self.today_str}.json"
        else:
            full_filename = f"{filename}.json"

        # Determinar la carpeta de salida
        if year:
            output_dir = self.base_output_dir / str(year)
        else:
            output_dir = self.base_output_dir

        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / full_filename

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(content, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error al guardar JSON en {path}: {e}", exc_info=True)

    def save_parquet(self, filename: str, content: list[dict], year: int = None):
        """
        Guarda contenido en formato Parquet.

        - Si se especifica 'year', guarda el archivo en 'data/landing/aemet/year=YYYY/'.
        - Si no se especifica 'year', guarda en 'data/landing/aemet/' directamente.

        Parámetros:
        -----------
        filename : str
            Nombre base del archivo (sin extensión).
        content : list[dict]
            Datos a guardar en el archivo Parquet.
        year : int, optional
            Año para particionar el almacenamiento en subcarpetas.
        """
        if year is not None:
            output_dir = self.base_output_dir / f"year={year}"
        else:
            output_dir = self.base_output_dir

        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{filename}.parquet"

        try:
            df = pd.DataFrame(content)
            df.to_parquet(path, index=False)
        except Exception as e:
            logger.error(f"[{self.name}] Error al guardar Parquet en {path}: {e}", exc_info=True)