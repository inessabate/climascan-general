from pathlib import Path
from datetime import datetime
import json

class BaseClient:
    def __init__(self, source_name: str):
        self.name = source_name.upper()
        self.output_dir = Path(__file__).resolve().parent.parent.parent.parent / "data" / "raw" / source_name.lower()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.today_str = datetime.today().strftime("%Y-%m-%d")

    def save_json(self, filename: str, content: dict | list, include_date: bool = True):
        """Saves JSON content to a file. By default, includes the current date in the filename."""
        if include_date:
            full_filename = f"{filename}_{self.today_str}.json"
        else:
            full_filename = f"{filename}.json"

        path = self.output_dir / full_filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2, ensure_ascii=False)
        print(f"[{self.name}] Saved: {path}")

    def log(self, message: str):
        print(f"[{self.name}] {message}")