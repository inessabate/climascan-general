import pandas as pd
from pathlib import Path

# Base del proyecto (subir un nivel desde src/)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

def main():
    # Ruta al dataset enriquecido final
    enriched_path = BASE_DIR / "data" / "aggregated" / "claims_classification.parquet"
    df = pd.read_parquet(str(enriched_path))

    print("=== INFO BÁSICA DEL DATASET ===")
    print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
    print("Columnas:", df.columns.tolist())
    print(df.head(), "\n")

    # --- Test 1: valores nulos por columna ---
    print("=== VALORES NULOS POR COLUMNA ===")
    print(df.isna().sum(), "\n")

    # --- Test 2: estadísticas básicas de variables meteorológicas ---
    meteo_vars = ["tmed", "tmin", "tmax", "prec", "hrmedia", "hrmax", "hrmin", "velmedia", "racha"]
    print("=== ESTADÍSTICAS DE VARIABLES METEOROLÓGICAS ===")
    print(df[meteo_vars].describe().T, "\n")

    # --- Test 3: distribución de la variable objetivo ---
    print("=== DISTRIBUCIÓN ocurre_siniestro ===")
    print(df["ocurre_siniestro"].value_counts(), "\n")

    # --- Test 4: cobertura espacial y temporal ---
    print("=== COBERTURA ESPACIAL Y TEMPORAL ===")
    print("Número de códigos postales únicos:", df["codigo_postal"].nunique())
    print("Rango de fechas:", df["fecha"].min(), "→", df["fecha"].max(), "\n")

    # --- Test 5: duplicados por (fecha, codigo_postal) ---
    print("=== DUPLICADOS POR (fecha, codigo_postal) ===")
    dup_count = df.duplicated(subset=["fecha", "codigo_postal"]).sum()
    if dup_count == 0:
        print("✅ No se han encontrado duplicados.")
    else:
        print(f"⚠️ Se han encontrado {dup_count} duplicados.")
        dups = df[df.duplicated(subset=["fecha", "codigo_postal"], keep=False)]
        print(dups.head(), "\n")

    # --- Test 6: ejemplos de siniestros ocurridos ---
    print("=== EJEMPLOS DE SINIESTROS ===")
    print(df[df["ocurre_siniestro"] == 1].head(10))

if __name__ == "__main__":
    main()
