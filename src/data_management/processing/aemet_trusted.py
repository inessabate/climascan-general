import logging
from pathlib import Path
import shutil
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from delta import configure_spark_with_delta_pip

# Config logger
logger = logging.getLogger(__name__)

# Reducir ruido de Spark y Py4J
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("pyspark").setLevel(logging.ERROR)

# Columnas candidatas a numéricas (en minúsculas tras A1)
NUM_COLS_CANDIDATES = [
    "tmed", "prec", "tmin", "tmax",
    "altitud", "velmedia", "racha",
    "presmax", "presmin",
    "hrmedia", "hrmax", "hrmin"
]

# Columnas a eliminar (A2) — en minúsculas tras A1
DROP_COLS_A2 = ["sol", "presmin", "presmax", "horapresmax", "horapresmin"]

# Periodo B2 (inclusive)
PERIOD_START = "2017-01-01"
PERIOD_END   = "2025-06-30"


def _count_and_log(df, msg):
    n = df.count()
    logger.info(f"{msg}: {n}")
    return n


def _null_counts_dict(df):
    exprs = [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns]
    row = df.select(*exprs).collect()[0]
    return row.asDict()


def _lowercase_columns(df):
    for c in df.columns:
        df = df.withColumnRenamed(c, c.lower())
    return df


def _normalize_numeric_and_prec(df):
    dtypes = dict(df.dtypes)
    ip_count, to_drop = 0, []

    if "prec" in dtypes:
        df = df.withColumn("prec", F.col("prec").cast("string"))
        ip_count = df.filter(F.col("prec") == "Ip").count()
        df = df.withColumn(
            "prec",
            F.when(F.col("prec") == "Ip", "0").otherwise(F.col("prec"))
        )
        df = df.withColumn("prec", F.regexp_replace("prec", ",", ".").cast("double"))

    for colname in NUM_COLS_CANDIDATES:
        if colname != "prec" and colname in dtypes:
            df = df.withColumn(colname, F.regexp_replace(F.col(colname), ",", ".").cast("double"))

    to_drop = [c for c in DROP_COLS_A2 if c in df.columns]
    if to_drop:
        df = df.drop(*to_drop)

    return df, ip_count, to_drop


def _ensure_fecha_as_date(df):
    if "fecha" not in df.columns:
        return df
    return df.withColumn("fecha", F.to_date("fecha"))


def main():
    try:
        builder = (
            SparkSession.builder
            .appName("Trusted_AEMET")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        )
        spark = configure_spark_with_delta_pip(builder).getOrCreate()
        logger.info("SparkSession inicializada con Delta.")

        # Paths
        base_path = Path(__file__).resolve().parents[3]
        landing_path = base_path / "data" / "landing" / "aemet_deltalake"
        trusted_path = base_path / "data" / "trusted" / "aemet_deltalake"

        if not landing_path.exists():
            raise FileNotFoundError(f"No existe la ruta de entrada: {landing_path}")

        # Lectura landing
        df = spark.read.format("delta").load(str(landing_path))
        total_inicial = _count_and_log(df, "Total inicial")
        nulls_initial = _null_counts_dict(df)

        # ===== A1 =====
        df = _lowercase_columns(df)

        # ===== A2 =====
        df, ip_count, dropped_cols = _normalize_numeric_and_prec(df)

        # ===== A4 =====
        if "codigo_postal" in df.columns:
            bad_cp_cnt = df.filter(
                (F.col("codigo_postal").isNotNull()) &
                (~F.col("codigo_postal").rlike(r"^(0[1-9]|[1-4][0-9]|5[0-2])"))
            ).count()

            df = df.filter(
                (F.col("codigo_postal").isNull()) |
                (F.col("codigo_postal").rlike(r"^(0[1-9]|[1-4][0-9]|5[0-2])"))
            )
        else:
            bad_cp_cnt = None

        # ===== B2 =====
        before_rows_b2 = df.count()
        stations_before = df.select("indicativo").distinct() if "indicativo" in df.columns else None
        df = _ensure_fecha_as_date(df)

        if "fecha" in df.columns:
            df = df.filter((F.col("fecha") >= PERIOD_START) & (F.col("fecha") <= PERIOD_END))

        rows_after_period = df.count()
        rows_removed_b2 = before_rows_b2 - rows_after_period

        if stations_before is not None:
            stations_after = df.select("indicativo").distinct()
            stations_removed_b2 = stations_before.join(stations_after, "indicativo", "left_anti").count()
        else:
            stations_removed_b2 = None

        # ===== C1 =====
        if all(c in df.columns for c in ["tmin", "tmax"]):
            bad_tmax_lt_tmin_cnt = df.filter(
                (F.col("tmax") < F.col("tmin")) & F.col("tmax").isNotNull() & F.col("tmin").isNotNull()
            ).count()
            df = df.filter(
                (F.col("tmax") >= F.col("tmin")) | F.col("tmax").isNull() | F.col("tmin").isNull()
            )
        else:
            bad_tmax_lt_tmin_cnt = None

        if all(c in df.columns for c in ["tmin", "tmax", "tmed"]):
            bad_tmed_range_cnt = df.filter(
                F.col("tmed").isNotNull() & F.col("tmin").isNotNull() & F.col("tmax").isNotNull() &
                ~((F.col("tmed") >= F.col("tmin")) & (F.col("tmed") <= F.col("tmax")))
            ).count()
            df = df.filter(
                (F.col("tmed").isNull()) |
                ((F.col("tmed") >= F.col("tmin")) & (F.col("tmed") <= F.col("tmax")))
            )
        else:
            bad_tmed_range_cnt = None

        # Totales finales
        total_final = _count_and_log(df, "Total final")
        nulls_final = _null_counts_dict(df)

        # Guardar trusted
        if trusted_path.exists():
            shutil.rmtree(trusted_path)
        df.write.format("delta").mode("overwrite").save(str(trusted_path))

        # ===== Log resumen =====
        logger.info("===== RESUMEN DQ =====")
        logger.info(f"A2 -> prec 'Ip' convertidos: {ip_count}, columnas eliminadas: {dropped_cols}")
        logger.info(f"A4 -> codigo_postal inválidos: {bad_cp_cnt}")
        logger.info(f"B2 -> filas fuera de periodo eliminadas: {rows_removed_b2}, estaciones eliminadas: {stations_removed_b2}")
        logger.info(f"C1 -> tmax<tmin: {bad_tmax_lt_tmin_cnt}, tmed fuera de rango: {bad_tmed_range_cnt}")
        logger.info(f"Nulos iniciales: {nulls_initial}")
        logger.info(f"Nulos finales: {nulls_final}")
        logger.info("===== FIN RESUMEN =====")

    except Exception:
        logger.exception("Error en capa trusted AEMET.")
        if __name__ == "__main__":
            sys.exit(1)
        else:
            raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()






