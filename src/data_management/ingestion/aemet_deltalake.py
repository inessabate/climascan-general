### NECESARIO JAVA 11
# Falta:
# - Implementar con logs
# - Añadirlo a los codigos main (de data management y el principal)
# - Proponer latex en vez de word


from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
from pathlib import Path
import shutil

# Crear SparkSession con soporte Delta Lake
builder = SparkSession.builder \
    .appName("Landing_AEMET_to_Delta") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")

spark = configure_spark_with_delta_pip(builder).getOrCreate()

# Definir rutas
base_path = Path(__file__).resolve().parents[3]  # Estamos en src/data_management/ingestion
landing_root = base_path / "data" / "landing"
aemet_json_root = landing_root / "aemet"
delta_output_path = landing_root / "aemet_deltalake"

# Eliminar carpeta de destino si ya existe (sobreescribe)
if delta_output_path.exists():
    shutil.rmtree(delta_output_path)

# Buscar todos los JSON dentro de subcarpetas por año
json_files = list(aemet_json_root.rglob("*.json"))

if not json_files:
    raise FileNotFoundError(f"No se encontraron archivos JSON en {aemet_json_root}")

# Leer todos los JSON con Spark (asumiendo mismo schema)
df = spark.read.option("multiLine", True).json([str(f) for f in json_files])

# Mostrar resumen
print(f"📦 Total de registros: {df.count()}")
print("🧪 Muestra de datos:")
df.show(10, truncate=False)

# Guardar como tabla Delta (sin procesar)
df.write.format("delta").mode("overwrite").save(str(delta_output_path))
print(f"✅ Datos guardados en formato Delta en: {delta_output_path}")
