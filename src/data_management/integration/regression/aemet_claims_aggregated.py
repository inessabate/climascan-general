#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import re
import sys
import shutil
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from delta import configure_spark_with_delta_pip

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("pyspark").setLevel(logging.ERROR)

_ILLEGAL_CHARS_PATTERN = re.compile(r"[ ,;{}\(\)\n\t=]+")


def _spark_session():
    builder = (
        SparkSession.builder
        .appName("Aggregated_AEMET_Claims_Parquet_to_Delta")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    logger.info("SparkSession inicializada con extensiones Delta.")
    return spark


def _sanitize_columns(df):
    mapping = {}
    for c in df.columns:
        clean = _ILLEGAL_CHARS_PATTERN.sub("_", c)
        clean = re.sub(r"_+", "_", clean).strip("_")
        if clean != c:
            mapping[c] = clean
    for src, dst in mapping.items():
        df = df.withColumnRenamed(src, dst)
    if mapping:
        logger.info(f"Columnas renombradas: {mapping}")
    return df


def _show_sample(df, n=10, truncate_chars=120):
    try:
        s = df._jdf.showString(n, truncate_chars, False)
        logger.info(f"Muestra (primeras {n} filas):\n{s}")
    except Exception:
        df.show(n, truncate=False)


def _convert_parquet_nanos_to_us(src_parquet: Path, tmp_parquet_dir: Path):
    """
    Convierte un parquet con TIMESTAMP(NANOS) a μs usando PyArrow,
    dejando un parquet 'equivalente' que Spark sí puede leer.
    """
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    if tmp_parquet_dir.exists():
        shutil.rmtree(tmp_parquet_dir)
    tmp_parquet_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Convirtiendo Parquet de ns → μs con PyArrow...")
    dataset = ds.dataset(str(src_parquet), format="parquet")
    # Escribimos forzando μs (allow_truncated_timestamps=True por si hay pérdida < μs)
    pq.write_table(
        dataset.to_table(),
        where=str(tmp_parquet_dir / "part-00000.parquet"),
        coerce_timestamps="us",
        allow_truncated_timestamps=True
    )
    logger.info(f"Parquet temporal (μs) escrito en: {tmp_parquet_dir}")


def main():
    try:
        spark = _spark_session()

        # Raíz del repo: .../climascan-general
        base_path = Path(__file__).resolve().parents[4]

        src_parquet = base_path / "data" / "aggregated" / "claims_regression.parquet"
        dst_delta = base_path / "data" / "aggregated" / "aemet_claims_deltalake"
        tmp_parquet_dir = base_path / "data" / "aggregated" / "_tmp_claims_regression_us"

        logger.info(f"Origen Parquet: {src_parquet}")
        logger.info(f"Destino Delta : {dst_delta}")

        if not src_parquet.exists():
            logger.error("No se encuentra el Parquet de entrada. Revisa la ruta.")
            sys.exit(1)

        # 1) Intento directo (si no hay ns, funcionará)
        try:
            df = spark.read.parquet(str(src_parquet))
        except Exception as e:
            msg = str(e)
            if "TIMESTAMP(NANOS" in msg or "Illegal Parquet type" in msg:
                logger.warning("Parquet con timestamps en ns detectado. Aplicando conversión a μs con PyArrow.")
                # 2) Fallback: convertir con PyArrow y reintentar lectura
                _convert_parquet_nanos_to_us(src_parquet, tmp_parquet_dir)
                df = spark.read.parquet(str(tmp_parquet_dir))
            else:
                raise  # otro error distinto

        df = _sanitize_columns(df)

        rowcount = df.count()
        logger.info(f"Registros leídos: {rowcount}")
        logger.info("Esquema post-sanitize:")
        df.printSchema()
        _show_sample(df, n=10)

        # Trazabilidad
        df = df.withColumn("etl_load_ts", F.current_timestamp())

        (
            df.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .save(str(dst_delta))
        )
        logger.info("Escritura Delta completada.")

        # Verificación
        df_check = spark.read.format("delta").load(str(dst_delta))
        out_count = df_check.count()
        logger.info(f"Registros en Delta: {out_count}")

        if out_count != rowcount:
            logger.warning(f"Recuento distinto: entrada={rowcount} salida={out_count}")

        # Limpieza temporal
        if tmp_parquet_dir.exists():
            shutil.rmtree(tmp_parquet_dir, ignore_errors=True)

        logger.info("Proceso finalizado OK.")

    except Exception:
        logger.exception("Error convirtiendo Parquet a Delta (aggregated).")
        sys.exit(2)


if __name__ == "__main__":
    main()

