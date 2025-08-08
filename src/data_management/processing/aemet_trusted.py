from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
from pyspark.sql.functions import col, regexp_replace
from pathlib import Path
import shutil

# 🚀 SparkSession
builder = SparkSession.builder \
    .appName("Trusted_AEMET") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
spark = configure_spark_with_delta_pip(builder).getOrCreate()

# 📁 Rutas
base_path = Path(__file__).resolve().parents[3]
input_path = base_path / "data" / "landing" / "aemet_deltalake"
trusted_path = base_path / "data" / "trusted" / "aemet_deltalake"

# 🧼 Sobreescribir si ya existe
if trusted_path.exists():
    shutil.rmtree(trusted_path)
    print(f"📁 Carpeta existente eliminada: {trusted_path}")

# 📥 Leer datos crudos desde Delta Lake (landing)
df = spark.read.format("delta").load(str(input_path))

# 🔍 Columnas numéricas que pueden tener comas como separador decimal
columnas_numericas = [
    "tmed", "prec", "tmin", "tmax",
    "velmedia", "racha",
    "presMax", "presMin",
    "hrMedia", "hrMax", "hrMin"
]

for colname in columnas_numericas:
    if colname in df.columns:
        df = df.withColumn(colname, regexp_replace(colname, ",", ".").cast("double"))

# ✅ Eliminar registros sin 'fecha' o 'indicativo'
df = df.dropna(subset=["fecha", "indicativo"])

# 📊 Ver resumen
print(f"✅ Total registros tras limpieza: {df.count()}")
print("🧪 Muestra de datos limpios:")
df.show(10, truncate=False)

# 💾 Guardar en formato Delta en zona "trusted"
df.write.format("delta").mode("overwrite").save(str(trusted_path))
print(f"🎉 Datos limpios guardados en: {trusted_path}")
