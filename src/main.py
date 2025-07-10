import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion.clients.aemet_client import AemetClient


def run_aemet():
    try:
        client = AemetClient()
        client.ejecutar()
    except Exception as e:
        print(f"Unexpected error in AEMET: {e}")

if __name__ == "__main__":
    run_aemet()
    print("AEMET data retrieval completed.")

