#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AEMET-Claims: Regresión de coste por siniestro (Delta aggregated).

Fixes frente a:
- GLM Gamma: "Sum of weights cannot be zero"
- GLM Tweedie: métricas enormes por inestabilidad numérica

Claves:
- features_std para GLM (StandardScaler withStd=True, withMean=False)
- weightCol='w' = 1.0 para GLM
- Winsorización suave de 'carga' en TRAIN (solo GLM) + mínimo positivo para Gamma
"""

import sys, math
from pathlib import Path

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import BooleanType
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.regression import GeneralizedLinearRegression, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder

# ========= Parámetros =========

ANIO_VALID = 2023
ADD_EXTREME_FLAGS = True        # flags p95 de clima
WINSORIZE_LABELS = True         # winsorización para GLM
WINSOR_Q = 0.999                # percentil superior para winsorizar 'carga' en TRAIN
GAMMA_MIN_POS = 1e-6            # mínimo positivo para Gamma

def repo_root_from_this_file():
    return Path(__file__).resolve().parents[3]

REPO_ROOT = repo_root_from_this_file()
DELTA_AGG_PATH = REPO_ROOT / "data" / "aggregated" / "aemet_claims_deltalake"
MODELS_DIR = REPO_ROOT / "models"
DIAG_DIR = MODELS_DIR / "diagnostics"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DIAG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH_PREP   = MODELS_DIR / "aemet_claims_prep_v2"
MODEL_PATH_GBT    = MODELS_DIR / "aemet_claims_reg_cost_gbt"
MODEL_PATH_TW     = MODELS_DIR / "aemet_claims_reg_cost_tweedie"
MODEL_PATH_GAMMA  = MODELS_DIR / "aemet_claims_reg_cost_gamma"

# ========= Spark =========

def build_spark():
    builder = (
        SparkSession.builder
        .appName("AEMET_Claims_Regression_All_Stable")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark

# ========= Carga & FE =========

def load_aggregated_delta(spark):
    if not DELTA_AGG_PATH.exists():
        raise FileNotFoundError(f"No existe Delta aggregated: {DELTA_AGG_PATH}")
    return spark.read.format("delta").load(str(DELTA_AGG_PATH))

def add_extreme_flags(df):
    qs = df.approxQuantile(["prec","tmax","racha"], [0.95], 100)
    p95_prec  = qs[0][0] if qs[0] else None
    p95_tmax  = qs[1][0] if qs[1] else None
    p95_racha = qs[2][0] if qs[2] else None
    if p95_prec is None or p95_tmax is None or p95_racha is None:
        return df
    return (df
            .withColumn("prec_ext",  F.when(F.col("prec")  >= F.lit(p95_prec), 1).otherwise(0))
            .withColumn("tmax_ext",  F.when(F.col("tmax")  >= F.lit(p95_tmax), 1).otherwise(0))
            .withColumn("racha_ext", F.when(F.col("racha") >= F.lit(p95_racha),1).otherwise(0))
           )

def feature_engineering(df):
    df = (df
          .withColumn("fecha_ocurrencia_ts", F.to_timestamp("fecha_ocurrencia"))
          .withColumn("anio", F.year("fecha_ocurrencia_ts"))
          .withColumn("mes",  F.month("fecha_ocurrencia_ts"))
          .withColumn("dow",  F.dayofweek("fecha_ocurrencia_ts"))
          .withColumn("log_carga", F.log1p(F.col("carga")))
         )
    if ADD_EXTREME_FLAGS:
        df = add_extreme_flags(df)

    num_cols = [
        "tmed","tmin","tmax","prec",
        "hrmedia","hrmax","hrmin",
        "velmedia","racha",
        "lat","lon",
        "mes","dow"
    ]
    if ADD_EXTREME_FLAGS:
        num_cols += ["prec_ext","tmax_ext","racha_ext"]

    existing_num_cols = [c for c in num_cols if c in df.columns]
    for c in existing_num_cols:
        df = df.withColumn(c, F.when(F.col(c).isNull(), F.lit(float("nan"))).otherwise(F.col(c)))

    # etiqueta válida
    df = df.filter(F.col("carga").isNotNull() & (F.col("carga") >= 0))
    return df, existing_num_cols

def temporal_split(df, anio_valid):
    return (df.filter(F.col("anio") < F.lit(anio_valid)),
            df.filter(F.col("anio") == F.lit(anio_valid)))

# ========= Prep (imputer+OHE+assembler) =========

def build_prep_pipeline(num_cols, cat_cols):
    imputer = Imputer(
        inputCols=num_cols,
        outputCols=[f"{c}_imp" for c in num_cols],
        strategy="median",
        missingValue=float("nan")
    )
    stages_cat, cat_out = [], []
    for c in cat_cols:
        idx = StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        oh  = OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_oh", handleInvalid="keep")
        stages_cat += [idx, oh]
        cat_out.append(f"{c}_oh")
    assembler = VectorAssembler(
        inputCols=[f"{c}_imp" for c in num_cols] + cat_out,
        outputCol="features_raw"
    )
    return Pipeline(stages=[imputer] + stages_cat + [assembler])

def vector_is_finite_udf():
    from pyspark.sql.types import BooleanType
    from pyspark.ml.linalg import DenseVector, SparseVector
    import math
    def _ok(v):
        if v is None: return False
        if isinstance(v, DenseVector):
            return all(math.isfinite(x) for x in v)
        else:  # SparseVector
            return all(math.isfinite(x) for x in v.values)
    return F.udf(_ok, BooleanType())

def prepare_datasets(train, valid, prep_pipeline, spark):
    prep_model = prep_pipeline.fit(train)
    train_prep = prep_model.transform(train)
    valid_prep = prep_model.transform(valid)

    # Filtrar vectores no finitos
    is_finite = vector_is_finite_udf()
    train_prep = train_prep.filter(is_finite("features_raw"))
    valid_prep = valid_prep.filter(is_finite("features_raw"))

    # Peso explícito = 1.0 para GLM
    train_prep = train_prep.withColumn("w", F.lit(1.0))
    valid_prep = valid_prep.withColumn("w", F.lit(1.0))

    # Estandarización SOLO para GLM (mantiene sparse: withStd=True, withMean=False)
    scaler = StandardScaler(inputCol="features_raw", outputCol="features_glm", withStd=True, withMean=False)
    scaler_model = scaler.fit(train_prep)
    train_prep = scaler_model.transform(train_prep)
    valid_prep = scaler_model.transform(valid_prep)

    return prep_model, train_prep, valid_prep

# ========= Etiqueta: winsor + mínimos =========

def winsorize_label(df, q=0.999, label="carga"):
    q_hi = df.approxQuantile(label, [q], 100)[0]
    return df.withColumn(label, F.when(F.col(label) > F.lit(q_hi), F.lit(q_hi)).otherwise(F.col(label))), q_hi

def enforce_gamma_min(df, label="carga", min_pos=1e-6):
    return df.withColumn(label, F.when(F.col(label) < F.lit(min_pos), F.lit(min_pos)).otherwise(F.col(label)))

# ========= Métricas & util =========

def compute_log_residual_var(model, df, label_col="log_carga", pred_col="prediction"):
    pred = model.transform(df).select(F.col(label_col).alias("y"), F.col(pred_col).alias("yhat"))
    res = pred.select((F.col("y") - F.col("yhat")).alias("e"))
    stats = res.agg(F.avg((F.col("e")**2)).alias("var")).collect()[0]
    return float(stats["var"]) if stats and stats["var"] is not None else 0.0

def extra_metrics(df, label="carga", pred="pred_carga"):
    eps = F.lit(1e-9)
    n = df.count()
    if n == 0: return {"n": 0}
    agg = df.agg(
        F.avg((F.col(pred)-F.col(label))**2).alias("mse"),
        F.avg(F.abs(F.col(pred)-F.col(label))).alias("mae"),
        F.avg(F.abs((F.col(pred)-F.col(label)) /
                    F.when(F.col(label)!=0, F.col(label)).otherwise(eps))).alias("mape"),
        F.corr(F.col(pred), F.col(label)).alias("corr"),
        F.var_samp(F.col(label)).alias("var_y")
    ).collect()[0]
    rmse = (agg["mse"] ** 0.5) if agg["mse"] is not None else float("nan")
    mae  = agg["mae"]
    mape = agg["mape"]*100 if agg["mape"] is not None else float("nan")
    r2   = 1 - (agg["mse"] / (agg["var_y"] if agg["var_y"] else float("inf")))
    return {"n": n, "RMSE": rmse, "MAE": mae, "MAPE%": mape, "R2": r2, "corr": agg["corr"]}

def decile_calibration(df, label="carga", pred="pred_carga"):
    from pyspark.sql.window import Window
    df2 = df.withColumn("rank", F.percent_rank().over(Window.orderBy(F.col(pred))))
    df2 = df2.withColumn("decile", (F.col("rank")*10).cast("int")+1)
    return (df2.groupBy("decile")
              .agg(F.avg(F.col(label)).alias("obs"),
                   F.avg(F.col(pred)).alias("pred"),
                   F.count("*").alias("n"))
              .orderBy("decile"))

def baseline_mean(train, valid, by_cols):
    mean_tbl = (train.groupBy(*by_cols).agg(F.avg("carga").alias("mean_carga")))
    joined   = (valid.join(mean_tbl, on=by_cols, how="left")
                     .fillna({"mean_carga": float(train.agg(F.avg("carga")).first()[0])})
                     .withColumnRenamed("mean_carga","pred_carga"))
    return joined

# ========= Modelos =========

def train_gbt_log(train_prep, valid_prep):
    gbt = GBTRegressor(featuresCol="features_raw", labelCol="log_carga", seed=42)
    paramGrid = (ParamGridBuilder()
                 .addGrid(gbt.maxDepth, [4,6,8])
                 .addGrid(gbt.maxIter,  [60,100,150])
                 .addGrid(gbt.stepSize, [0.05, 0.1, 0.2])
                 .build())
    eval_log = RegressionEvaluator(labelCol="log_carga", predictionCol="prediction", metricName="rmse")
    tvs = TrainValidationSplit(estimator=gbt, estimatorParamMaps=paramGrid,
                               evaluator=eval_log, trainRatio=0.8, seed=42)
    tv_model = tvs.fit(train_prep)
    best_gbt = tv_model.bestModel

    log_var = compute_log_residual_var(best_gbt, train_prep, label_col="log_carga", pred_col="prediction")
    pred = (best_gbt.transform(valid_prep)
            .withColumn("pred_carga", F.exp(F.col("prediction") + F.lit(0.5*log_var)) - F.lit(1.0)))
    return best_gbt, pred

def train_glm_tweedie(train_prep, valid_prep):
    # Winsorizar etiqueta en TRAIN (suaviza colas para IRLS)
    tw_train = train_prep
    if WINSORIZE_LABELS:
        tw_train, q_hi = winsorize_label(tw_train, WINSOR_Q, "carga")
        print(f"[Tweedie] Winsor aplicado en TRAIN: p{int(WINSOR_Q*1000)/10}% = {q_hi:.2f}")

    tw = GeneralizedLinearRegression(
        featuresCol="features_glm",
        labelCol="carga",
        weightCol="w",
        family="tweedie",
        variancePower=1.5,   # prueba 1.3–1.7 si quieres afinar
        linkPower=0.0,       # log link
        maxIter=200,
        regParam=1e-3,       # un poco más de regularización para estabilidad
        tol=1e-6
    )
    model = tw.fit(tw_train)
    pred  = model.transform(valid_prep).withColumn("pred_carga", F.col("prediction"))
    return model, pred

def train_glm_gamma(train_prep, valid_prep):
    gm_train = train_prep.filter(F.col("carga") > 0)
    gm_valid = valid_prep.filter(F.col("carga") > 0)

    if WINSORIZE_LABELS:
        gm_train, q_hi = winsorize_label(gm_train, WINSOR_Q, "carga")
        print(f"[Gamma] Winsor aplicado en TRAIN: p{int(WINSOR_Q*1000)/10}% = {q_hi:.2f}")

    gm_train = enforce_gamma_min(gm_train, "carga", GAMMA_MIN_POS)

    n_pos = gm_train.count()
    if n_pos == 0:
        raise RuntimeError("No hay positivos en TRAIN para Gamma tras filtros.")

    gm = GeneralizedLinearRegression(
        featuresCol="features_glm",
        labelCol="carga",
        weightCol="w",
        family="gamma",
        link="log",
        maxIter=200,
        regParam=1e-3,   # regularización para estabilidad
        tol=1e-6
    )
    model = gm.fit(gm_train)
    pred  = model.transform(gm_valid).withColumn("pred_carga", F.col("prediction"))
    return model, pred, gm_valid

# ========= Main =========

def main():
    spark = build_spark()

    # 1) Carga
    df = load_aggregated_delta(spark)

    # 2) FE
    df, num_cols = feature_engineering(df)
    cat_cols = ["lob", "segmento_cliente_detalle"]

    # 3) Split
    train, valid = temporal_split(df, ANIO_VALID)
    n_train, n_valid = train.count(), valid.count()
    print(f"Train: {n_train:,} | Valid: {n_valid:,} (Año valid = {ANIO_VALID})")
    if n_train == 0 or n_valid == 0:
        print("Advertencia: split vacío. Revisa ANIO_VALID/histórico.")
        sys.exit(1)

    # 4) Prep + scaler GLM + pesos
    prep_pipeline = build_prep_pipeline(num_cols, cat_cols)
    prep_model, train_prep, valid_prep = prepare_datasets(train, valid, prep_pipeline, spark)

    # 5) Evaluadores
    evaluator_rmse = RegressionEvaluator(labelCol="carga", predictionCol="pred_carga", metricName="rmse")
    evaluator_mae  = RegressionEvaluator(labelCol="carga", predictionCol="pred_carga", metricName="mae")

    # 6) Modelos
    best_gbt, pred_gbt = train_gbt_log(train_prep, valid_prep)
    rmse_gbt = evaluator_rmse.evaluate(pred_gbt); mae_gbt = evaluator_mae.evaluate(pred_gbt)

    try:
        model_tw, pred_tw = train_glm_tweedie(train_prep, valid_prep)
        rmse_tw = evaluator_rmse.evaluate(pred_tw); mae_tw = evaluator_mae.evaluate(pred_tw)
    except Exception as e:
        print(f"[AVISO] Tweedie falló: {e}")
        model_tw, pred_tw, rmse_tw, mae_tw = None, None, float("nan"), float("nan")

    try:
        model_gm, pred_gm, valid_pos = train_glm_gamma(train_prep, valid_prep)
        rmse_gm = evaluator_rmse.evaluate(pred_gm); mae_gm = evaluator_mae.evaluate(pred_gm)
    except Exception as e:
        print(f"[AVISO] Gamma falló: {e}")
        model_gm, pred_gm, rmse_gm, mae_gm, valid_pos = None, None, float("nan"), float("nan"), None

    # 7) Resultados
    print("\nResultados en VALID (euros):")
    print(f"GBT (tuned + bias-corr) → RMSE = {rmse_gbt:,.2f} | MAE = {mae_gbt:,.2f}")
    print(f"Tweedie GLM             → RMSE = {rmse_tw:,.2f} | MAE = {mae_tw:,.2f}")
    print(f"Gamma GLM  (>0)         → RMSE = {rmse_gm:,.2f} | MAE = {mae_gm:,.2f}  (evaluado en y>0)")

    # 8) Métricas extra + deciles
    print("\n=== Métricas extra (GBT) ==="); print(extra_metrics(pred_gbt, "carga", "pred_carga"))
    cal_gbt = decile_calibration(pred_gbt, "carga", "pred_carga")
    (cal_gbt.coalesce(1).write.mode("overwrite").option("header", True)
        .csv(str(DIAG_DIR / "calibration_deciles_gbt_valid")))

    if pred_tw is not None:
        print("\n=== Métricas extra (Tweedie) ==="); print(extra_metrics(pred_tw, "carga", "pred_carga"))
        cal_tw = decile_calibration(pred_tw, "carga", "pred_carga")
        (cal_tw.coalesce(1).write.mode("overwrite").option("header", True)
            .csv(str(DIAG_DIR / "calibration_deciles_tweedie_valid")))

    if pred_gm is not None:
        print("\n=== Métricas extra (Gamma, y>0) ==="); print(extra_metrics(pred_gm, "carga", "pred_carga"))
        cal_gm = decile_calibration(pred_gm, "carga", "pred_carga")
        (cal_gm.coalesce(1).write.mode("overwrite").option("header", True)
            .csv(str(DIAG_DIR / "calibration_deciles_gamma_valid_pos")))

    # 9) Baselines
    base1 = baseline_mean(train, valid, ["lob"])
    base2 = baseline_mean(train, valid, ["lob","segmento_cliente_detalle"])
    print("\n=== Baseline lob ===");          print(extra_metrics(base1, "carga", "pred_carga"))
    print("=== Baseline lob×segmento ==="); print(extra_metrics(base2, "carga", "pred_carga"))

    # 10) Guardado
    prep_model.write().overwrite().save(str(MODEL_PATH_PREP))
    best_gbt.write().overwrite().save(str(MODEL_PATH_GBT))
    if model_tw is not None: model_tw.write().overwrite().save(str(MODEL_PATH_TW))
    if model_gm is not None: model_gm.write().overwrite().save(str(MODEL_PATH_GAMMA))
    print(f"\nModelos guardados en:\n- {MODEL_PATH_PREP}\n- {MODEL_PATH_GBT}\n- {MODEL_PATH_TW}\n- {MODEL_PATH_GAMMA}")

    spark.stop()

if __name__ == "__main__":
    main()


