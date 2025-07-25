from pathlib import Path
from datetime import datetime
import json
import logging

# Logger del módulo
logger = logging.getLogger(__name__)

class BaseClient:
    def __init__(self, source_name: str):
        self.name = source_name.upper()
        self.output_dir = (
            Path(__file__).resolve().parent.parent.parent.parent
            / "data"
            / "landing"
            / source_name.lower()
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.today_str = datetime.today().strftime("%Y-%m-%d")

    def save_json(self, filename: str, content: dict | list, include_date: bool = True):
        """
        Guarda contenido JSON en un archivo.
        Por defecto incluye la fecha actual en el nombre del archivo.
        """
        if include_date:
            full_filename = f"{filename}_{self.today_str}.json"
        else:
            full_filename = f"{filename}.json"

        path = self.output_dir / full_filename

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(content, f, indent=2, ensure_ascii=False)
            logger.info(f"[{self.name}] JSON guardado correctamente en {path}")
        except Exception as e:
            logger.error(f"[{self.name}] Error al guardar JSON en {path}: {e}", exc_info=True)