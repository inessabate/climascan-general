# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import shutil
import sys
import tempfile
import re

from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("pyspark").setLevel(logging.ERROR)

_ILLEGAL_CHARS_PATTERN = re.compile(r"[ ,;{}\(\)\n\t=]+")

def _show_sample(df, n=10, truncate_chars=100):
    try:
        s = df._jdf.showString(n, truncate_chars, False)
        logger.info(f"Muestra (primeras {n} filas):\n{s}")
    except Exception:
        df.show(n, truncate=False)

def _spark_session():
    builder = (
        SparkSession.builder
        .appName("Trusted_Claims_Parquet_to_Delta")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()

def _convert_parquet_to_us_with_pyarrow(src_path: Path, dst_path: Path):
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    logger.info("Convirtiendo Parquet a microsegundos con PyArrow (coerce_timestamps='us')...")
    dataset = ds.dataset(str(src_path), format="parquet")
    table = dataset.to_table()
    tmp_dir = tempfile.mkdtemp(prefix="claims_parquet_us_")
    tmp_parquet = Path(tmp_dir) / "repacked.parquet"
    pq.write_table(
        table,
        where=str(tmp_parquet),
        coerce_timestamps="us",
        allow_truncated_timestamps=True,
        version="2.6"
    )
    if dst_path.exists():
        shutil.rmtree(dst_path)
    dst_path.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp_parquet), str(dst_path / "part-00000.parquet"))
    logger.info(f"Parquet reconvertido (μs) en: {dst_path}")

def _sanitize_column_names(df):
    """
    - minúsculas
    - reemplaza caracteres ilegales [espacios, , ; { } ( ) \n \t =] por '_'
    - colapsa múltiples '_' y quita '_' extremos
    - evita duplicados añadiendo sufijos _1, _2, ...
    """
    original = df.columns
    cleaned = []
    seen = set()
    mapping = {}
    for c in original:
        nc = c.lower()
        nc = _ILLEGAL_CHARS_PATTERN.sub("_", nc)
        nc = re.sub(r"_+", "_", nc).strip("_")
        if nc == "":
            nc = "col"
        base = nc
        k = 1
        while nc in seen:
            k += 1
            nc = f"{base}_{k}"
        seen.add(nc)
        cleaned.append(nc)
        if nc != c:
            mapping[c] = nc
    if mapping:
        logger.info(f"Renombrando columnas (sanitización): {mapping}")
    return df.toDF(*cleaned)

def main():
    try:
        spark = _spark_session()

        base_path = Path(__file__).resolve().parents[3]
        src_parquet = base_path / "data" / "trusted" / "claims" / "weather_claims.parquet"
        dst_delta   = base_path / "data" / "trusted" / "claims" / "claims_deltalake"

        logger.info(f"Origen Parquet: {src_parquet}")
        logger.info(f"Destino Delta : {dst_delta}")

        if not src_parquet.exists():
            raise FileNotFoundError(f"No existe el Parquet origen: {src_parquet}")

        # 1) Intento directo con Spark (puede fallar por TIMESTAMP(NANOS))
        try:
            df = spark.read.parquet(str(src_parquet))
        except Exception as e:
            msg = str(e)
            if "TIMESTAMP(NANOS" in msg or "Illegal Parquet type: INT64 (TIMESTAMP(NANOS" in msg:
                fixed_dir = src_parquet.parent / "weather_claims_us_tmp"
                _convert_parquet_to_us_with_pyarrow(src_parquet, fixed_dir)
                df = spark.read.parquet(str(fixed_dir))
            else:
                raise

        logger.info(f"Filas leídas: {df.count()}")
        df.printSchema()
        _show_sample(df, 10)

        # 2) Sanitizar nombres de columnas antes de escribir a Delta
        df = _sanitize_column_names(df)

        # 3) Escribir como Delta (limpiando destino antes)
        if dst_delta.exists():
            shutil.rmtree(dst_delta)
        df.write.format("delta").mode("overwrite").save(str(dst_delta))
        logger.info("Conversión completada. Delta guardado correctamente.")

        # Limpieza temporal si existiera
        tmp_dir = src_parquet.parent / "weather_claims_us_tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    except Exception:
        logger.exception("Error convirtiendo Parquet a Delta (claims).")
        if __name__ == "__main__":
            sys.exit(1)
        else:
            raise

if __name__ == "__main__":
    main()


