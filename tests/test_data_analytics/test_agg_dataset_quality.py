import pandas as pd
from pathlib import Path

# Base del proyecto (subir un nivel desde src/)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

def main():
    # Ruta al dataset enriquecido final
    enriched_path = BASE_DIR / "data" / "aggregated" / "claims_regression.parquet"
    df = pd.read_parquet(str(enriched_path))

    print("=== INFO BÁSICA DEL DATASET ===")
    print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
    print("Columnas:", df.columns.tolist())
    print(df.head(), "\n")

    # --- Test 1: valores nulos por columna ---
    print("=== VALORES NULOS POR COLUMNA ===")
    print(df.isna().sum(), "\n")

    # --- Test 2: cuántos siniestros tienen todas las variables meteo nulas ---
    meteo_vars = ["tmed", "tmin", "tmax", "prec", "hrmedia", "hrmax", "hrmin", "velmedia", "racha"]
    null_rows = df[meteo_vars].isna().all(axis=1).sum()
    print(f"Siniestros sin datos meteo: {null_rows} ({null_rows/len(df)*100:.2f}%)\n")

    # --- Test 3: estadísticas básicas de variables meteorológicas ---
    print("=== ESTADÍSTICAS DE VARIABLES METEOROLÓGICAS ===")
    print(df[meteo_vars].describe().T, "\n")

    # --- Test 4: códigos postales únicos ---
    print("Número de códigos postales únicos en claims:", df["codigo_postal_norm"].nunique(), "\n")

    # --- Test 5: ejemplo particular ---
    test_hash = "54e7229a9d3ff9c7"
    test_claim = df[df["siniestro_hash"] == test_hash]
    print(f"=== Ejemplo de siniestro {test_hash} ===")
    print(test_claim)

if __name__ == "__main__":
    main()
