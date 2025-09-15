### NECESARIO JAVA 11
# Falta:
# - Añadirlo a los codigos main (de data management y el principal)
# - Proponer latex en vez de word

import logging
import re
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
from pathlib import Path
import shutil
import sys

logger = logging.getLogger(__name__)

YEAR_DIR_REGEX = re.compile(r"^year=\d{4}$")

def main():
    try:
        # Spark + Delta
        builder = SparkSession.builder \
            .appName("Landing_AEMET_Parquet_to_Delta") \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        spark = configure_spark_with_delta_pip(builder).getOrCreate()
        logger.info("SparkSession inicializada con extensiones Delta.")

        # Rutas
        base_path = Path(__file__).resolve().parents[3]
        landing_root = base_path / "data" / "landing"
        aemet_root = landing_root / "aemet"          # aquí hay year=YYYY y quizá ficheros sueltos
        delta_output_path = landing_root / "aemet_deltalake"

        logger.info(f"Ruta base: {base_path}")
        logger.info(f"Entrada (root): {aemet_root}")
        logger.info(f"Salida Delta: {delta_output_path}")

        if not aemet_root.exists():
            raise FileNotFoundError(f"No existe la ruta de entrada: {aemet_root}")

        # Detectar SOLO subcarpetas con nombre year=YYYY
        year_dirs = [p for p in aemet_root.iterdir() if p.is_dir() and YEAR_DIR_REGEX.match(p.name)]
        year_dirs = sorted(year_dirs)  # opcional, por orden
        if not year_dirs:
            raise FileNotFoundError(f"No se encontraron subcarpetas 'year=YYYY' dentro de {aemet_root}")

        logger.info(f"Subcarpetas válidas detectadas: {[d.name for d in year_dirs]}")

        # Limpiar destino
        if delta_output_path.exists():
            shutil.rmtree(delta_output_path)
            logger.info(f"Carpeta de salida existente eliminada: {delta_output_path}")

        # Leer únicamente parquet dentro de las carpetas year=YYYY
        df = spark.read \
            .option("mergeSchema", "true") \
            .option("basePath", str(aemet_root)) \
            .parquet(str(aemet_root / "year=*/"))

        logger.info("Lectura de Parquet completada desde subcarpetas 'year=YYYY'.")
        logger.info(f"Columnas detectadas: {df.columns}")

        # === LOGGING DE REGISTROS ===
        record_count = df.count()
        logger.info(f"Total de registros cargados: {record_count}")

        # Por partición year (si existe)
        if "year" in df.columns:
            by_year_df = df.groupBy("year").count().orderBy("year")
            counts_by_year = [f"{row['year']}={row['count']}" for row in by_year_df.collect()]
            logger.info("Registros por year: " + ", ".join(counts_by_year))
        else:
            logger.warning("No se encontró la columna de partición 'year' en el DataFrame.")

        # Muestra rápida
        try:
            sample_str = df._jdf.showString(10, 100, False)
            logger.info("Muestra de registros (hasta 10 filas):\n" + sample_str)
        except Exception:
            logger.warning("No se pudo formatear la muestra con showString; usando df.show() en STDOUT.")
            df.show(10, truncate=False)

        # Escritura Delta
        df.write.format("delta").mode("overwrite").save(str(delta_output_path))
        logger.info(f"Datos guardados en formato Delta en: {delta_output_path}")

        # Verificación leyendo desde Delta
        df_delta = spark.read.format("delta").load(str(delta_output_path))
        written_count = df_delta.count()
        logger.info(f"Total de registros verificados en Delta: {written_count}")

        if "year" in df_delta.columns:
            by_year_written = df_delta.groupBy("year").count().orderBy("year")
            counts_written = [f"{row['year']}={row['count']}" for row in by_year_written.collect()]
            logger.info("Registros por year en Delta: " + ", ".join(counts_written))

        # Comparación simple
        if written_count != record_count:
            logger.warning(f"Diferencia de conteos (antes={record_count}, después={written_count}).")
        else:
            logger.info("Verificación OK: el número de registros coincide antes y después de escribir.")

    except Exception as e:
        logger.exception("Error durante la ingesta Parquet→Delta (filtrando year=YYYY).")
        if __name__ == "__main__":
            sys.exit(1)
        else:
            raise e

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()


