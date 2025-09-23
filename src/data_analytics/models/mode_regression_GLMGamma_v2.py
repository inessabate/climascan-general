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
from pyspark.ml.feature import (
    Imputer, StringIndexer, OneHotEncoder, VectorAssembler
)
from pyspark.ml.regression import GeneralizedLinearRegression, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator


# =====================
# Parámetros principales
# =====================

# Año que se usará como VALIDACIÓN (entrenas con años anteriores)
ANIO_VALID = 2023

# Rutas relativas al repo (ajústalas si tu árbol difiere)
def repo_root_from_this_file():
    # Este script está en: .../src/data_analytics/models/modelo_regresion_GLMGamma.py
    # la raíz del repo es parents[3]
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


# =====================
# Carga y Feature Engineering
# =====================

def load_aggregated_delta(spark):
    if not DELTA_AGG_PATH.exists():
        raise FileNotFoundError(f"No existe el Delta de aggregated en: {DELTA_AGG_PATH}")
    return spark.read.format("delta").load(str(DELTA_AGG_PATH))


def feature_engineering(df):
    """
    - Deriva variables temporales desde fecha_ocurrencia (timestamp_ntz).
    - Crea log_carga = log1p(carga).
    - Normaliza NULL -> NaN en columnas numéricas disponibles (para que Imputer funcione).
    - Filtra etiqueta válida (carga >= 0) para el conjunto base.
    """
    # 1) Derivar columnas temporales y log de la etiqueta
    df = (df
          .withColumn("fecha_ocurrencia_ts", F.to_timestamp("fecha_ocurrencia"))
          .withColumn("anio", F.year("fecha_ocurrencia_ts"))
          .withColumn("mes", F.month("fecha_ocurrencia_ts"))
          .withColumn("dow", F.dayofweek("fecha_ocurrencia_ts"))
          .withColumn("log_carga", F.log1p(F.col("carga")))
          )

    # 2) Lista de numéricas candidatas (algunas podrían no estar en el DF)
    num_cols = [
        # meteorología
        "tmed", "tmin", "tmax", "prec",
        "hrmedia", "hrmax", "hrmin",
        "velmedia", "racha",
        # localización
        "lat", "lon",
        # calendario
        "mes", "dow"
    ]

    # 3) Convierte NULL -> NaN SOLO en las columnas que existan
    existing_num_cols = [c for c in num_cols if c in df.columns]
    for c in existing_num_cols:
        df = df.withColumn(c, F.when(F.col(c).isNull(), F.lit(float("nan"))).otherwise(F.col(c)))

    # 4) Etiqueta válida (>= 0 para el conjunto base; GLM filtrará > 0 más adelante)
    df = df.filter(F.col("carga").isNotNull() & (F.col("carga") >= 0))

    return df, existing_num_cols  # devolvemos df y la lista efectiva de numéricas


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
    """
    Construye dos pipelines:
      - GLM Gamma (label 'carga')
      - Gradient Boosted Trees (label 'log_carga')
    Importante: el assembler usa SOLO columnas imputadas *_imp y categóricas OHE.
    """

    # 1) Imputación de numéricas -> genera *_imp
    imputer = Imputer(
        inputCols=num_cols,
        outputCols=[f"{c}_imp" for c in num_cols],
        strategy="median",
        missingValue=float("nan")  # imputar NaN
    )

    # 2) Categóricas: index + one-hot (con nulos a 'KEEP')
    stages_cat, cat_out = [], []
    for c in cat_cols:
        idx = StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        oh = OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_oh", handleInvalid="keep")
        stages_cat += [idx, oh]
        cat_out.append(f"{c}_oh")

    # 3) Ensamble: SOLO imputadas + OHE
    feature_cols = [f"{c}_imp" for c in num_cols] + cat_out
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")

    # 4) Modelos
    glm = GeneralizedLinearRegression(
        featuresCol="features_raw",
        labelCol="carga",
        family="gamma",
        link="log",
        maxIter=100,
        regParam=0.1  # L2 (única disponible en GLM)
    )

    gbt = GBTRegressor(
        featuresCol="features_raw",
        labelCol="log_carga",
        maxIter=200,
        maxDepth=6,
        stepSize=0.05,
        subsamplingRate=0.8
    )

    # 5) Pipelines completos
    pipeline_glm = Pipeline(stages=[imputer] + stages_cat + [assembler, glm])
    pipeline_gbt = Pipeline(stages=[imputer] + stages_cat + [assembler, gbt])

    return pipeline_glm, pipeline_gbt


# =====================
# Entrenamiento y evaluación
# =====================

def train_and_evaluate(train, valid, pipeline_glm, pipeline_gbt):
    # Datasets base sin nulos en label
    train_base = train.filter(F.col("carga").isNotNull())
    valid_base = valid.filter(F.col("carga").isNotNull())

    # GLM Gamma: y > 0
    train_glm = train_base.filter(F.col("carga") > 0)
    valid_glm = valid_base.filter(F.col("carga") > 0)

    evaluator_rmse = RegressionEvaluator(labelCol="carga", predictionCol="pred_carga", metricName="rmse")
    evaluator_mae  = RegressionEvaluator(labelCol="carga", predictionCol="pred_carga", metricName="mae")

    # ---- GLM con try/fallback ----
    model_glm = None
    rmse_glm = mae_glm = float("nan")
    try:
        model_glm = pipeline_glm.fit(train_glm)
        pred_glm  = model_glm.transform(valid_glm).withColumn("pred_carga", F.col("prediction"))
        rmse_glm  = evaluator_rmse.evaluate(pred_glm)
        mae_glm   = evaluator_mae.evaluate(pred_glm)
    except Exception as e:
        print(f"[AVISO] GLM Gamma no ha podido entrenar: {e}")
        print("         Continuamos con GBT; revisa nulos/extremos/constantes y la distribución de 'carga'.")

    # ---- GBT (log1p) ----
    model_gbt = pipeline_gbt.fit(train_base)
    pred_gbt  = (model_gbt.transform(valid_base)
                 .withColumn("pred_carga", F.exp(F.col("prediction")) - F.lit(1.0)))
    rmse_gbt  = evaluator_rmse.evaluate(pred_gbt)
    mae_gbt   = evaluator_mae.evaluate(pred_gbt)

    return (model_glm, rmse_glm, mae_glm), (model_gbt, rmse_gbt, mae_gbt)


# =====================
# Main
# =====================

def main():
    spark = build_spark()

    # 1) Carga
    df = load_aggregated_delta(spark)

    # 2) Feature engineering (devuelve df y numéricas efectivas)
    df, num_cols = feature_engineering(df)

    # 3) Categóricas
    cat_cols = ["lob", "segmento_cliente_detalle"]

    # 4) Split temporal
    train, valid = temporal_split(df, ANIO_VALID)

    # Chequeo rápido de tamaños
    n_train = train.count()
    n_valid = valid.count()
    print(f"Train: {n_train:,} filas | Valid: {n_valid:,} filas (Año valid = {ANIO_VALID})")
    if n_train == 0 or n_valid == 0:
        print("Advertencia: alguno de los splits está vacío. Revisa ANIO_VALID o tu histórico.")
        spark.stop()
        sys.exit(1)

    # 5) Pipelines
    pipeline_glm, pipeline_gbt = build_pipelines(num_cols, cat_cols)

    # 6) Entrenamiento y evaluación
    (model_glm, rmse_glm, mae_glm), (model_gbt, rmse_gbt, mae_gbt) = train_and_evaluate(
        train, valid, pipeline_glm, pipeline_gbt
    )

    print("\nResultados en VALID (euros):")
    if model_glm is not None:
        print(f"GLM Gamma     → RMSE = {rmse_glm:,.2f} | MAE = {mae_glm:,.2f}")
    else:
        print("GLM Gamma     → no entrenado (ver aviso arriba)")
    print(f"GBT log(carga)→ RMSE = {rmse_gbt:,.2f} | MAE = {mae_gbt:,.2f}")

    # 7) Guardado de modelos
    if model_glm is not None:
        model_glm.write().overwrite().save(str(MODEL_PATH_GLM))
    model_gbt.write().overwrite().save(str(MODEL_PATH_GBT))
    print("\nModelos guardados en:")
    if model_glm is not None:
        print(f"- {MODEL_PATH_GLM}")
    print(f"- {MODEL_PATH_GBT}")

    spark.stop()


if __name__ == "__main__":
    main()
