from src.ingestion.clients.aemet_client import AemetClient

if __name__ == '__main__':
    cliente = AemetClient()
    cliente.execute_aemet()


