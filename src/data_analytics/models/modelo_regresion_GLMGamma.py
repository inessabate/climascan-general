#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Entrenamiento de modelos de regresión de coste por siniestro (capa aggregated en Delta).

Flujo:
1) Carga Delta: data/aggregated/aemet_claims_deltalake/
2) Feature engineering: variables temporales (mes, dow), log1p(carga)
3) Split temporal (train vs valid) por año de fecha_ocurrencia
4) Modelos:
   - GLM Gamma (link log) sobre 'carga'
   - GBTRegressor sobre 'log_carga' (luego invertimos log para evaluar en euros)
5) Evaluación: RMSE y MAE en el conjunto de validación
6) Persistencia de modelos en ./models/
"""

import sys
from pathlib import Path

from delta import configure_spark_with_delta_pip

from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.regression import GeneralizedLinearRegression, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator


# =====================
# Parámetros principales
# =====================

# Año que se usará como VALIDACIÓN (entrenas con años anteriores)
ANIO_VALID = 2023

# Rutas relativas al repo (ajústalas si tu árbol difiere)
def repo_root_from_this_file():
    # Si este script está en: .../src/data_analytics/models/entrenar_modelos.py
    # la raíz del repo es parents[4]
    return Path(__file__).resolve().parents[3]

REPO_ROOT = repo_root_from_this_file()
DELTA_AGG_PATH = REPO_ROOT / "data" / "aggregated" / "aemet_claims_deltalake"
MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH_GLM = MODELS_DIR / "aemet_claims_reg_cost_glm"
MODEL_PATH_GBT = MODELS_DIR / "aemet_claims_reg_cost_gbt"


# =====================
# Spark Session
# =====================

def build_spark():
    builder = (
        SparkSession.builder
        .appName("AEMET_Claims_Regression_Training")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")  # opcional
    return spark

# COMPROBACION QUE EXTENSON ESTA ACTIVA Y QUE SE PUEDE LEER LA TABLA DELTA
from pathlib import Path

spark = build_spark()
print("Spark version:", spark.version)
print("Extensions:", spark.conf.get("spark.sql.extensions"))
print("Catalog   :", spark.conf.get("spark.sql.catalog.spark_catalog"))

# Calcula raíz del repo en función de dónde está este .py
REPO_ROOT = Path(__file__).resolve().parents[3]  # porque lo tienes en src/data_analytics/models
DELTA_AGG_PATH = REPO_ROOT / "data" / "aggregated" / "aemet_claims_deltalake"

df = spark.read.format("delta").load(str(DELTA_AGG_PATH))
df.printSchema()
df.limit(5).show(truncate=False)


# =====================
# Carga y Feature Engineering
# =====================

def load_aggregated_delta(spark):
    if not DELTA_AGG_PATH.exists():
        raise FileNotFoundError(f"No existe el Delta de aggregated en: {DELTA_AGG_PATH}")
    df = spark.read.format("delta").load(str(DELTA_AGG_PATH))
    return df


def feature_engineering(df):
    """
    - Convierte fecha_ocurrencia (timestamp_ntz) a timestamp y deriva variables temporales.
    - Crea log_carga = log1p(carga) para el modelo de GBT.
    """
    df = (df
          .withColumn("fecha_ocurrencia_ts", F.to_timestamp("fecha_ocurrencia"))
          .withColumn("anio", F.year("fecha_ocurrencia_ts"))
          .withColumn("mes", F.month("fecha_ocurrencia_ts"))
          .withColumn("dow", F.dayofweek("fecha_ocurrencia_ts"))
          .withColumn("log_carga", F.log1p(F.col("carga")))
          )
    return df


def temporal_split(df, anio_valid):
    """
    Split temporal: train = años < anio_valid, valid = anio == anio_valid
    """
    train = df.filter(F.col("anio") < F.lit(anio_valid))
    valid = df.filter(F.col("anio") == F.lit(anio_valid))
    return train, valid


# =====================
# Modelos y Pipelines
# =====================

def build_pipelines(num_cols, cat_cols):
    # Indexación y One-Hot de categóricas
    indexers = [StringIndexer(handleInvalid="keep", inputCol=c, outputCol=f"{c}_idx") for c in cat_cols]
    encoders = [OneHotEncoder(handleInvalid="keep",
                              inputCols=[f"{c}_idx"],
                              outputCols=[f"{c}_oh"]) for c in cat_cols]

    # Ensamble de features
    feature_cols = num_cols + [f"{c}_oh" for c in cat_cols]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    scaler = StandardScaler(withMean=True, withStd=True, inputCol="features_raw", outputCol="features")

    # Modelo 1: GLM Gamma sobre 'carga'
    glm = GeneralizedLinearRegression(
        featuresCol="features",
        labelCol="carga",
        family="gamma",
        link="log",
        maxIter=100,
        regParam=0.01
    )
    pipeline_glm = Pipeline(stages=[*indexers, *encoders, assembler, scaler, glm])

    # Modelo 2: GBT sobre 'log_carga'
    gbt = GBTRegressor(
        featuresCol="features",
        labelCol="log_carga",
        maxDepth=6,
        maxIter=200,
        stepSize=0.05,
        subsamplingRate=0.8
    )
    pipeline_gbt = Pipeline(stages=[*indexers, *encoders, assembler, scaler, gbt])

    return pipeline_glm, pipeline_gbt


# =====================
# Entrenamiento y evaluación
# =====================

def train_and_evaluate(train, valid, pipeline_glm, pipeline_gbt):
    # Entrenar GLM
    model_glm = pipeline_glm.fit(train)
    pred_glm = model_glm.transform(valid).withColumn("pred_carga", F.col("prediction"))

    # Entrenar GBT (sobre log_carga). Invertimos log1p para evaluar en euros.
    model_gbt = pipeline_gbt.fit(train)
    pred_gbt = (model_gbt.transform(valid)
                .withColumn("pred_carga", F.exp(F.col("prediction")) - F.lit(1.0)))

    # Evaluadores (en euros)
    evaluator_rmse = RegressionEvaluator(labelCol="carga", predictionCol="pred_carga", metricName="rmse")
    evaluator_mae = RegressionEvaluator(labelCol="carga", predictionCol="pred_carga", metricName="mae")

    rmse_glm = evaluator_rmse.evaluate(pred_glm)
    mae_glm = evaluator_mae.evaluate(pred_glm)

    rmse_gbt = evaluator_rmse.evaluate(pred_gbt)
    mae_gbt = evaluator_mae.evaluate(pred_gbt)

    return (model_glm, rmse_glm, mae_glm), (model_gbt, rmse_gbt, mae_gbt)


# =====================
# Main
# =====================

def main():
    spark = build_spark()

    # 1) Carga
    df = load_aggregated_delta(spark)

    # 2) Feature engineering
    df = feature_engineering(df)

    # 3) Selección de columnas útiles
    #    Numéricas: meteorología + lat/lon + temporales
    num_cols = [
        "tmed", "tmin", "tmax", "prec",
        "hrmedia", "hrmax", "hrmin",
        "velmedia", "racha",
        "lat", "lon",
        "mes", "dow"
    ]
    #    Categóricas: LOB y segmento
    cat_cols = ["lob", "segmento_cliente_detalle"]

    # 4) Split temporal
    train, valid = temporal_split(df, ANIO_VALID)

    # Chequeo rápido de tamaños
    n_train = train.count()
    n_valid = valid.count()
    print(f"Train: {n_train:,} filas | Valid: {n_valid:,} filas (Año valid = {ANIO_VALID})")
    if n_train == 0 or n_valid == 0:
        print("Advertencia: alguno de los splits está vacío. Revisa ANIO_VALID o tu histórico.")
        sys.exit(1)

    # 5) Pipelines
    pipeline_glm, pipeline_gbt = build_pipelines(num_cols, cat_cols)

    # 6) Entrenamiento y evaluación
    (model_glm, rmse_glm, mae_glm), (model_gbt, rmse_gbt, mae_gbt) = train_and_evaluate(
        train, valid, pipeline_glm, pipeline_gbt
    )

    print("\nResultados en VALID (euros):")
    print(f"GLM Gamma     → RMSE = {rmse_glm:,.2f} | MAE = {mae_glm:,.2f}")
    print(f"GBT log(carga)→ RMSE = {rmse_gbt:,.2f} | MAE = {mae_gbt:,.2f}")

    # 7) Guardado de modelos
    model_glm.write().overwrite().save(str(MODEL_PATH_GLM))
    model_gbt.write().overwrite().save(str(MODEL_PATH_GBT))
    print(f"\nModelos guardados en:\n- {MODEL_PATH_GLM}\n- {MODEL_PATH_GBT}")

    spark.stop()


if __name__ == "__main__":
    main()
