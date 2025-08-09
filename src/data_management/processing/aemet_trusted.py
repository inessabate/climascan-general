import logging
from pathlib import Path
import shutil
import sys
import csv

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from delta import configure_spark_with_delta_pip

logger = logging.getLogger(__name__)

# Columnas candidatas a numéricas
NUM_COLS_CANDIDATES = [
    "tmed", "prec", "tmin", "tmax",
    "altitud", "velmedia", "racha",
    "presMax", "presMin",
    "hrMedia", "hrMax", "hrMin"
]
# Clave lógica mínima
KEY_COLS = ["fecha", "indicativo"]


def _normalize_numeric_columns(df):
    """Convierte columnas numéricas a double, manejando comas como separador decimal."""
    dtypes = dict(df.dtypes)
    for colname in NUM_COLS_CANDIDATES:
        if colname in dtypes:
            if dtypes[colname] == "string":
                df = df.withColumn(colname, F.regexp_replace(F.col(colname), ",", ".").cast("double"))
            else:
                df = df.withColumn(colname, F.col(colname).cast("double"))
    return df


def _count_and_log(df, msg):
    n = df.count()
    logger.info(f"{msg}: {n}")
    return n


def _log_null_counts(df):
    """Devuelve un diccionario con el conteo de nulos."""
    exprs = [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns]
    row = df.select(*exprs).collect()[0]
    return row.asDict()


def _show_sample(df, n=10, truncate_chars=100):
    try:
        s = df._jdf.showString(n, truncate_chars, False)
        logger.info(f"Muestra (primeras {n} filas):\n{s}")
    except Exception:
        logger.warning("No se pudo formatear la muestra con showString; usando df.show() en STDOUT.")
        df.show(n, truncate=False)


def main():
    try:
        # Spark con Delta
        builder = (
            SparkSession.builder
            .appName("Trusted_AEMET")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        )
        spark = configure_spark_with_delta_pip(builder).getOrCreate()
        logger.info("SparkSession inicializada con Delta.")

        # Rutas
        base_path = Path(__file__).resolve().parents[3]
        landing_path = base_path / "data" / "landing" / "aemet_deltalake"
        trusted_path = base_path / "data" / "trusted" / "aemet_deltalake"
        dq_report_path = base_path / "data" / "trusted" / "aemet_dq_report"

        logger.info(f"Entrada (Delta landing): {landing_path}")
        logger.info(f"Salida  (Delta trusted): {trusted_path}")
        logger.info(f"Reporte DQ (CSV): {dq_report_path}")

        if not landing_path.exists():
            raise FileNotFoundError(f"No existe la ruta de entrada: {landing_path}")

        # Lectura landing
        df = spark.read.format("delta").load(str(landing_path))
        total_inicial = _count_and_log(df, "Total de registros leídos (landing)")
        logger.info(f"Columnas: {df.columns}")
        df.printSchema()
        _show_sample(df, 10)

        # Diccionario para el reporte
        dq_report = {
            "total_inicial": total_inicial,
            "nulos_inicial": _log_null_counts(df),
            "duplicados": None,
            "filas_eliminadas": 0
        }

        # Duplicados
        if all(k in df.columns for k in KEY_COLS):
            dups = df.groupBy(*KEY_COLS).count().filter(F.col("count") > 1)
            dq_report["duplicados"] = dups.count()
        else:
            dq_report["duplicados"] = "No evaluado"

        # Normalización numérica
        df = _normalize_numeric_columns(df)

        # Reglas de Data Quality
        removed_total = 0

        # 1) Claves mínimas
        before = df.count()
        df = df.dropna(subset=[c for c in KEY_COLS if c in df.columns])
        removed_total += before - df.count()

        # 2) Precipitación >= 0
        if "prec" in df.columns:
            before = df.count()
            df = df.filter((F.col("prec").isNull()) | (F.col("prec") >= 0.0))
            removed_total += before - df.count()

        # 3) tmin <= tmax
        if "tmin" in df.columns and "tmax" in df.columns:
            before = df.count()
            df = df.filter(
                (F.col("tmin").isNull()) | (F.col("tmax").isNull()) | (F.col("tmin") <= F.col("tmax"))
            )
            removed_total += before - df.count()

        # 4) Humedades [0, 100]
        for hcol in ["hrMin", "hrMedia", "hrMax"]:
            if hcol in df.columns:
                before = df.count()
                df = df.filter(
                    (F.col(hcol).isNull()) | ((F.col(hcol) >= 0.0) & (F.col(hcol) <= 100.0))
                )
                removed_total += before - df.count()

        # 5) Altitud [-100, 4000]
        if "altitud" in df.columns:
            before = df.count()
            df = df.filter(
                (F.col("altitud").isNull()) | ((F.col("altitud") >= -100.0) & (F.col("altitud") <= 4000.0))
            )
            removed_total += before - df.count()

        dq_report["filas_eliminadas"] = removed_total
        dq_report["total_final"] = df.count()
        dq_report["nulos_final"] = _log_null_counts(df)

        # Guardar trusted
        if trusted_path.exists():
            shutil.rmtree(trusted_path)
        df.write.format("delta").mode("overwrite").save(str(trusted_path))
        logger.info(f"Trusted guardado en: {trusted_path}")

        # ======== Guardar reporte CSV fijo (sin Spark) ========
        if dq_report_path.exists():
            shutil.rmtree(dq_report_path)
        dq_report_path.mkdir(parents=True, exist_ok=True)

        csv_path = dq_report_path / "dq_report.csv"

        # Aplanamos el dict
        flat_report = []
        for k, v in dq_report.items():
            if isinstance(v, dict):
                for subk, subv in v.items():
                    flat_report.append((f"{k}.{subk}", str(subv)))
            else:
                flat_report.append((k, str(v)))

        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerows(flat_report)

        logger.info(f"Reporte de calidad guardado en: {csv_path}")
        # ======================================================

    except Exception:
        logger.exception("Error en capa trusted AEMET.")
        if __name__ == "__main__":
            sys.exit(1)
        else:
            raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()



