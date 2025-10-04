#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AEMET-Claims: Regresión de coste por siniestro (Delta aggregated)

Incluye:
- FE + imputación + OHE (para GBT) y vector GLM separado (numéricas + target encodings, sin OHE)
- GBT global (log1p + corrección lognormal/bias)
- Tweedie GLM robusto (escala, piso en y/scale, offset log, regularización, clip)
- (Opcional) Gamma GLM robusto con fallback lognormal (desactivado por defecto)
- Two-part: Frecuencia (LR) × Severidad (GBT en positivos con corrección lognormal y clip)
- Baselines globales (lob y lob×segmento) + blends (RMSE y MAE), blend por deciles, fallback top-1%
- Entrenamiento per-LOB (reusa hiperparámetros del GBT global), selección de ganador corregida
- Métricas limpias + CSV + calibración por deciles
"""

import sys, math
from pathlib import Path
from datetime import datetime
from functools import reduce

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import BooleanType
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import GeneralizedLinearRegression, GBTRegressor
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder

# =====================
# Parámetros principales
# =====================

ANIO_VALID = 2023

# Flags / robustecedores
ADD_EXTREME_FLAGS = True
WINSORIZE_LABELS = True
WINSOR_Q = 0.999
GAMMA_MIN_POS = 1e-6
TWEEDIE_MIN_POS = 1e-6
USE_TWO_PART = True
USE_PER_LOB = True
USE_GAMMA_MODELS = False   # Gamma suele rendir peor; activarlo solo para comparar

# GLM: pisos/offsets/estabilidad
GLM_SCALED_Y_MIN = 1e-2       # piso para y/scale
GLM_USE_OFFSET   = True       # offset constante = log(media de y/scale)

# Tweedie: rejilla conservadora
TWEEDIE_VGRID = (1.1, 1.3, 1.5)

# Deciles
DECILES_REPARTITION = 200

# Rutas
def repo_root_from_this_file():
    return Path(__file__).resolve().parents[3]

REPO_ROOT = repo_root_from_this_file()
DELTA_AGG_PATH = REPO_ROOT / "data" / "aggregated" / "aemet_claims_deltalake"
MODELS_DIR = REPO_ROOT / "models"
DIAG_DIR = MODELS_DIR / "diagnostics"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DIAG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH_PREP   = MODELS_DIR / "aemet_claims_prep_v3"
MODEL_PATH_GBT    = MODELS_DIR / "aemet_claims_reg_cost_gbt"
MODEL_PATH_TW     = MODELS_DIR / "aemet_claims_reg_cost_tweedie"
MODEL_PATH_GAMMA  = MODELS_DIR / "aemet_claims_reg_cost_gamma"
MODEL_PATH_LR_POS = MODELS_DIR / "aemet_claims_two_part_lr"
MODEL_PATH_SEV_GBT_POS = MODELS_DIR / "aemet_claims_two_part_sev_gbt"

# =====================
# Spark (silenciar logs)
# =====================

def build_spark():
    builder = (
        SparkSession.builder
        .appName("AEMET_Claims_Regression_All")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.ui.showConsoleProgress", "false")
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    try:
        log4j = spark._jvm.org.apache.log4j
        log4j.LogManager.getRootLogger().setLevel(log4j.Level.ERROR)
    except Exception:
        pass
    return spark

# =====================
# Carga y Feature Engineering
# =====================

def load_aggregated_delta(spark):
    if not DELTA_AGG_PATH.exists():
        raise FileNotFoundError(f"No existe Delta aggregated: {DELTA_AGG_PATH}")
    return spark.read.format("delta").load(str(DELTA_AGG_PATH))

def add_extreme_flags(df):
    qs = df.approxQuantile(["prec","tmax","racha"], [0.95], 0.001)
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
    df = df.withColumn("carga", F.col("carga").cast("double"))
    df = (df
          .withColumn("fecha_ocurrencia_ts", F.to_timestamp("fecha_ocurrencia"))
          .withColumn("anio", F.year("fecha_ocurrencia_ts"))
          .withColumn("mes",  F.month("fecha_ocurrencia_ts"))
          .withColumn("dow",  F.dayofweek("fecha_ocurrencia_ts"))
         )
    num_cols = [
        "tmed","tmin","tmax","prec",
        "hrmedia","hrmax","hrmin",
        "velmedia","racha",
        "lat","lon",
        "mes","dow"
    ]
    for c in [c for c in num_cols if c in df.columns]:
        df = df.withColumn(c, F.col(c).cast("double"))

    if ADD_EXTREME_FLAGS:
        df = add_extreme_flags(df)
        num_cols += ["prec_ext","tmax_ext","racha_ext"]

    existing_num_cols = [c for c in num_cols if c in df.columns]
    for c in existing_num_cols:
        df = df.withColumn(c, F.when(F.col(c).isNull(), F.lit(float("nan"))).otherwise(F.col(c)))

    df = df.filter(F.col("carga").isNotNull() & (F.col("carga") >= 0))
    df = df.withColumn("log_carga", F.log1p(F.col("carga")))
    return df, existing_num_cols

def temporal_split(df, anio_valid):
    train = df.filter(F.col("anio") < F.lit(anio_valid))
    valid = df.filter(F.col("anio") == F.lit(anio_valid))
    return train, valid

# =====================
# Auditoría etiqueta
# =====================

def quick_label_audit(df, name):
    df = df.withColumn("carga", F.col("carga").cast("double"))
    probs = [0.0, 0.5, 0.9, 0.99, 0.999]
    qs = df.approxQuantile("carga", probs, 0.001)
    n  = df.count()
    print(f"[{name}] n={n:,} | p0={qs[0]:.6f} p50={qs[1]:.6f} p90={qs[2]:.6f} p99={qs[3]:.6f} p99.9={qs[4]:.6f}")

# =====================
# Target Encoding (sin leakage)
# =====================

def add_target_encoding(train, valid, by_cols, label="carga", name="te"):
    tbl = (train.groupBy(*by_cols).agg(F.avg(F.col(label)).alias(f"{name}_mean")))
    train2 = train.join(tbl, on=by_cols, how="left")
    valid2 = valid.join(tbl, on=by_cols, how="left")
    global_mean = float(train.agg(F.avg(F.col(label))).first()[0])
    train2 = train2.fillna({f"{name}_mean": global_mean})
    valid2 = valid2.fillna({f"{name}_mean": global_mean})
    return train2, valid2

# =====================
# Preparación (fit en TRAIN)
# =====================

def build_prep_pipeline(num_cols, cat_cols, glm_extra_cols=None):
    """
    Crea DOS vectores:
      - features_raw: numéricas imputadas + OHE (para GBT)
      - features_glm: numéricas imputadas + columnas de TE_*_mean (sin OHE) → más estable en GLM
    """
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

    assembler_raw = VectorAssembler(
        inputCols=[f"{c}_imp" for c in num_cols] + cat_out,
        outputCol="features_raw"
    )

    glm_extra_cols = glm_extra_cols or []
    assembler_glm = VectorAssembler(
        inputCols=[f"{c}_imp" for c in num_cols] + glm_extra_cols,
        outputCol="features_glm"
    )

    return Pipeline(stages=[imputer] + stages_cat + [assembler_raw, assembler_glm])

def vector_is_finite_udf():
    from pyspark.ml.linalg import DenseVector, SparseVector
    import math
    def _ok(v):
        if v is None: return False
        if isinstance(v, DenseVector):
            return all(math.isfinite(x) for x in v)
        else:
            return all(math.isfinite(x) for x in v.values)
    return F.udf(_ok, BooleanType())

def vector_has_nnz_udf():
    from pyspark.ml.linalg import DenseVector, SparseVector
    def _nnz(v):
        if v is None: return False
        if isinstance(v, DenseVector):
            return any(x != 0.0 for x in v)
        else:
            return v.numNonzeros() > 0
    return F.udf(_nnz, BooleanType())

def prepare_datasets(train, valid, prep_pipeline):
    prep_model = prep_pipeline.fit(train)
    train_prep = prep_model.transform(train)
    valid_prep = prep_model.transform(valid)

    is_finite = vector_is_finite_udf()
    has_nnz   = vector_has_nnz_udf()

    train_prep = (train_prep
                  .filter(is_finite("features_raw")).filter(has_nnz("features_raw"))
                  .filter(is_finite("features_glm")).filter(has_nnz("features_glm")))
    valid_prep = (valid_prep
                  .filter(is_finite("features_raw")).filter(has_nnz("features_raw"))
                  .filter(is_finite("features_glm")).filter(has_nnz("features_glm")))

    return prep_model, train_prep, valid_prep

# =====================
# Métricas & utilidades
# =====================

def compute_log_residual_var(model, df, label_col="log_carga", pred_col="prediction"):
    pred = model.transform(df).select(F.col(label_col).alias("y"), F.col(pred_col).alias("yhat"))
    res = pred.select((F.col("y") - F.col("yhat")).alias("e"))
    stats = res.agg(F.avg((F.col("e")**2)).alias("var")).collect()[0]
    return float(stats["var"]) if stats and stats["var"] is not None else 0.0

def extra_metrics_smape(df, label="carga", pred="pred_carga"):
    eps = F.lit(1e-9)
    n = df.count()
    if n == 0: return {"n": 0}
    agg = df.agg(
        F.avg((F.col(pred)-F.col(label))**2).alias("mse"),
        F.avg(F.abs(F.col(pred)-F.col(label))).alias("mae"),
        (F.avg(2*F.abs(F.col(pred)-F.col(label)) /
               (F.abs(F.col(pred)) + F.abs(F.col(label)) + eps))).alias("smape"),
        F.corr(F.col(pred), F.col(label)).alias("corr"),
        F.var_samp(F.col(label)).alias("var_y")
    ).collect()[0]
    rmse = (agg["mse"] ** 0.5) if agg["mse"] is not None else float("nan")
    mae  = agg["mae"]
    smape = 100*agg["smape"] if agg["smape"] is not None else float("nan")
    r2   = 1 - (agg["mse"] / (agg["var_y"] if agg["var_y"] else float("inf")))
    return {"n": n, "RMSE": rmse, "MAE": mae, "SMAPE%": smape, "R2": r2, "corr": agg["corr"]}

# --- Curva por deciles (definida a nivel de módulo, única) ---
from pyspark.sql.window import Window
def decile_calibration(df, label="carga", pred="pred_carga", ntiles=10, seed=13):
    """
    Para cada decil de la predicción, devuelve media observada (obs), media predicha (pred) y n.
    Rompe empates con jitter determinista para evitar colisiones.
    """
    df2 = (
        df.select(
            F.col(label).cast("double").alias(label),
            F.col(pred).cast("double").alias(pred)
        )
        .where(F.col(label).isNotNull() & F.col(pred).isNotNull())
        .withColumn("_j", F.rand(seed))
    )
    w = Window.orderBy(F.col(pred).asc(), F.col("_j").asc())
    out = (
        df2.withColumn("decile", F.ntile(ntiles).over(w))
           .groupBy("decile")
           .agg(
               F.avg(F.col(label)).alias("obs"),
               F.avg(F.col(pred)).alias("pred"),
               F.count(F.lit(1)).alias("n")
           )
           .orderBy("decile")
    )
    return out

def baseline_mean(train, valid, by_cols):
    mean_tbl = (train.groupBy(*by_cols).agg(F.avg("carga").alias("mean_carga")))
    joined   = (valid.join(mean_tbl, on=by_cols, how="left")
                     .fillna({ "mean_carga": float(train.agg(F.avg("carga")).first()[0]) })
                     .withColumnRenamed("mean_carga","pred_carga"))
    return joined

def winsorize_label(df, q=0.999, label="carga"):
    q_hi = df.approxQuantile(label, [float(q)], 0.001)[0]
    return df.withColumn(label, F.when(F.col(label) > F.lit(q_hi), F.lit(q_hi)).otherwise(F.col(label))), q_hi

def enforce_gamma_min(df, label="carga", min_pos=1e-6):
    return df.withColumn(label, F.when(F.col(label) < F.lit(min_pos), F.lit(min_pos)).otherwise(F.col(label)))

def compute_label_scale(df, label='carga', q=0.99):
    qs = df.approxQuantile(label, [float(q)], 0.001)
    s = float(qs[0]) if qs and qs[0] and qs[0] > 0 else 1.0
    s = max(min(s, 1e7), 100.0)
    print(f"[GLM] Escala de etiqueta (q={q}): {s:,.4f}")
    return s

def cap_predictions_by_label_quantile(train_df, pred_df, label="carga", pred_col="pred_carga", q=0.99, factor=1.0):
    cap = float(train_df.approxQuantile(label, [float(q)], 0.001)[0]) * factor
    cap = max(cap, 100.0)
    return (pred_df
            .withColumn(pred_col, F.when(F.col(pred_col) < 0, F.lit(0.0))
                                    .otherwise(F.col(pred_col)))
            .withColumn(pred_col, F.when(F.col(pred_col) > F.lit(cap), F.lit(cap))
                                    .otherwise(F.col(pred_col))))

def ensure_constant_offset(df, colname, value):
    return df.withColumn(colname, F.lit(float(value)))

# ===== Resultados: recolector, resumen y CSV =====

def init_results():
    return []

def add_result(results, model_name, pred_df, label="carga", pred_col="pred_carga", extra_fn=None):
    m = extra_metrics_smape(pred_df, label, pred_col)
    row = {
        "model": model_name,
        "n": int(m["n"]),
        "RMSE": float(m["RMSE"]) if m["RMSE"] is not None else None,
        "MAE": float(m["MAE"]) if m["MAE"] is not None else None,
        "SMAPE%": float(m["SMAPE%"]) if m["SMAPE%"] is not None else None,
        "R2": float(m["R2"]) if m["R2"] is not None else None,
        "corr": float(m["corr"]) if m["corr"] is not None else None
    }
    if extra_fn is not None:
        row.update(extra_fn())
    results.append(row)

def flush_results(results, spark, out_dir):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dfres = spark.createDataFrame(results)
    outpath = str(out_dir / f"metrics_valid_{ts}")
    (dfres.coalesce(1).write.mode("overwrite").option("header", True).csv(outpath))
    print("\n===== RESUMEN MÉTRICAS (VALID) =====")
    for r in results:
        print(f"{r['model']}: n={r['n']:,} | RMSE={r['RMSE']:.2f} | MAE={r['MAE']:.2f} | SMAPE%={r['SMAPE%']:.2f} | R2={r['R2']:.4f}")
    print(f"(CSV guardado en: {outpath})")
    return outpath

# =====================
# BLENDS AVANZADOS
# =====================

def prepare_join_for_blend(pred_df, base_df, key_col="siniestro_hash"):
    left = base_df.select(key_col, "carga", F.col("pred_carga").alias("pred_base"))
    right = pred_df.select(key_col, F.col("pred_carga").alias("pred_model"))
    return left.join(right, key_col, "inner")

def tune_blend_lambda_for_join(join_df, metric="rmse"):
    pred_col = "pred_carga"
    label_col = "carga"
    evaluator = RegressionEvaluator(labelCol=label_col, predictionCol=pred_col,
                                    metricName=("rmse" if metric=="rmse" else "mae"))
    grid = [i/20.0 for i in range(0, 21)]  # 0.00 ... 1.00
    best_lam, best_val, best_df = 0.0, float("inf"), None
    for lam in grid:
        b = join_df.withColumn(pred_col,
                               F.lit(lam)*F.col("pred_model") + (1.0-F.lit(lam))*F.col("pred_base"))
        val = evaluator.evaluate(b)
        if val < best_val:
            best_lam, best_val, best_df = lam, val, b
    return best_lam, best_df

def blend_predictions(pred_df, base_df, key_col="siniestro_hash", lam=0.5):
    left = base_df.select(key_col, "carga", F.col("pred_carga").alias("pred_base"))
    right = pred_df.select(key_col, F.col("pred_carga").alias("pred_model"))
    return (left.join(right, key_col, "inner")
                .withColumn("pred_carga", F.lit(lam)*F.col("pred_model") + (1.0-F.lit(lam))*F.col("pred_base")))

def tune_blend_lambda(pred_df, base_df, metric="rmse", key_col="siniestro_hash"):
    evaluator = RegressionEvaluator(labelCol="carga", predictionCol="pred_carga",
                                    metricName=("rmse" if metric=="rmse" else "mae"))
    grid = [i/20.0 for i in range(0, 21)]  # 0.00 ... 1.00
    best_lam, best_val, best_blend = 0.0, float("inf"), None
    for lam in grid:
        b = blend_predictions(pred_df, base_df, key_col=key_col, lam=lam)
        val = evaluator.evaluate(b)
        if val < best_val:
            best_lam, best_val, best_blend = lam, val, b
    return best_lam, best_blend

def tune_blend_by_decile(pred_df, base_df, metric="rmse", key_col="siniestro_hash", num_deciles=10):
    joined = prepare_join_for_blend(pred_df, base_df, key_col=key_col)
    from pyspark.sql.window import Window
    w = Window.orderBy(F.col("pred_model"))
    joined = joined.withColumn("rank", F.percent_rank().over(w))
    joined = joined.withColumn("decile", (F.col("rank")*num_deciles).cast("int") + F.lit(1))

    lam_map = {}
    pieces = []
    for d in range(1, num_deciles+1):
        part = joined.filter(F.col("decile") == d)
        if part.limit(1).count() == 0:
            continue
        lam_d, b_d = tune_blend_lambda_for_join(part, metric=metric)
        lam_map[d] = lam_d
        pieces.append(b_d.select(key_col, "carga", F.col("pred_carga")))
    if not pieces:
        raise RuntimeError("No se pudo construir blend por deciles (sin piezas).")
    blended = reduce(lambda a,b: a.unionByName(b), pieces)
    return blended, lam_map

def apply_top_percentile_fallback(pred_df, base_df, key_col="siniestro_hash", top_p=0.01, pred_col="pred_carga"):
    thr = float(pred_df.approxQuantile(pred_col, [1.0 - top_p], 0.001)[0])
    left = base_df.select(key_col, "carga", F.col("pred_carga").alias("pred_base"))
    right = pred_df.select(key_col, F.col(pred_col).alias("pred_model"))
    joined = left.join(right, key_col, "inner")
    out = (joined
           .withColumn("pred_carga",
                       F.when(F.col("pred_model") >= F.lit(thr), F.col("pred_base"))
                        .otherwise(F.col("pred_model")))
           .select(key_col, "carga", "pred_carga"))
    return out, thr

# =====================
# Modelos (globales)
# =====================

def train_gbt_log(train_prep, valid_prep, tune=True, base_params=None, parallelism=1):
    if tune:
        gbt = GBTRegressor(featuresCol="features_raw", labelCol="log_carga", seed=42)
        paramGrid = (ParamGridBuilder()
                     .addGrid(gbt.maxDepth, [4, 6, 8])
                     .addGrid(gbt.maxIter,  [60, 100, 150])
                     .addGrid(gbt.stepSize, [0.05, 0.1, 0.2])
                     .build())
        eval_log = RegressionEvaluator(labelCol="log_carga", predictionCol="prediction", metricName="rmse")
        tvs = TrainValidationSplit(estimator=gbt,
                                   estimatorParamMaps=paramGrid,
                                   evaluator=eval_log,
                                   trainRatio=0.8,
                                   seed=42)
        tvs.setParallelism(parallelism)
        tv_model = tvs.fit(train_prep)
        best = tv_model.bestModel
        best_params = {
            "maxDepth": best.getMaxDepth(),
            "maxIter":  best.getMaxIter(),
            "stepSize": best.getStepSize()
        }
    else:
        if base_params is None:
            base_params = {"maxDepth": 6, "maxIter": 100, "stepSize": 0.1}
        best = GBTRegressor(
            featuresCol="features_raw", labelCol="log_carga", seed=42,
            maxDepth=base_params["maxDepth"],
            maxIter=base_params["maxIter"],
            stepSize=base_params["stepSize"]
        ).fit(train_prep)
        best_params = dict(base_params)

    log_var = compute_log_residual_var(best, train_prep, label_col="log_carga", pred_col="prediction")
    pred = (best.transform(valid_prep)
            .withColumn("pred_carga", F.exp(F.col("prediction") + F.lit(0.5*log_var)) - F.lit(1.0)))
    return best, pred, best_params

def tune_tweedie(train_prep, valid_prep, vgrid=TWEEDIE_VGRID):
    tr = train_prep.withColumn(
        "carga_tw",
        F.when(F.col("carga") <= F.lit(TWEEDIE_MIN_POS), F.lit(TWEEDIE_MIN_POS)).otherwise(F.col("carga"))
    )
    if WINSORIZE_LABELS:
        tr, q_hi = winsorize_label(tr, WINSOR_Q, "carga_tw")
        print(f"[Tweedie] Winsor TRAIN p{int(WINSOR_Q*1000)/10}% = {q_hi:.6f}")

    scale = compute_label_scale(tr, label="carga_tw", q=0.99)
    tr = tr.withColumn("carga_tw_s_raw", F.col("carga_tw")/F.lit(scale))
    tr = tr.withColumn("carga_tw_s", F.when(F.col("carga_tw_s_raw") < F.lit(GLM_SCALED_Y_MIN),
                                            F.lit(GLM_SCALED_Y_MIN)).otherwise(F.col("carga_tw_s_raw")))

    if GLM_USE_OFFSET:
        mu0 = float(tr.agg(F.avg("carga_tw_s")).first()[0] or 1.0)
        off_val = math.log(max(mu0, GLM_SCALED_Y_MIN))
        tr = ensure_constant_offset(tr, "offset_tw", off_val)

    is_finite = vector_is_finite_udf()
    has_nnz   = vector_has_nnz_udf()
    tr = tr.filter(is_finite("features_glm")).filter(has_nnz("features_glm"))
    va = valid_prep.filter(is_finite("features_glm")).filter(has_nnz("features_glm"))
    if GLM_USE_OFFSET:
        va = ensure_constant_offset(va, "offset_tw", off_val)

    n_tr = tr.count()
    if n_tr == 0:
        raise RuntimeError("TRAIN para Tweedie quedó vacío tras filtros.")

    fit_cols = ["features_glm", "carga_tw_s"] + (["offset_tw"] if GLM_USE_OFFSET else [])
    tr_fit = tr.select(*fit_cols)
    va_pred = va

    best_m, best_pred, best_mae, best_v = None, None, float("inf"), None
    evaluator_mae = RegressionEvaluator(labelCol="carga", predictionCol="pred_carga", metricName="mae")

    for v in vgrid:
        tw = GeneralizedLinearRegression(
            featuresCol="features_glm",
            labelCol="carga_tw_s",
            family="tweedie",
            variancePower=v,
            linkPower=0.0,
            predictionCol="pred_tw_s",
            maxIter=200, regParam=1e-1, tol=1e-6,
            offsetCol=("offset_tw" if GLM_USE_OFFSET else None)
        )
        m = tw.fit(tr_fit)
        p = (m.transform(va_pred)
               .withColumn("pred_carga", F.col("pred_tw_s")*F.lit(scale)))
        p = cap_predictions_by_label_quantile(train_prep, p, label="carga", pred_col="pred_carga", q=0.99, factor=1.0)
        mae = evaluator_mae.evaluate(p)
        print(f"[Tweedie v={v}] MAE={mae:,.4f}")
        if mae < best_mae:
            best_m, best_pred, best_mae, best_v = m, p, mae, v

    print(f"[Tweedie] Mejor variancePower={best_v} | MAE={best_mae:,.4f}")
    return best_m, best_pred, best_v

def train_glm_gamma(train_prep, valid_prep):
    gm_train = train_prep.filter(F.col("carga") > 0)
    gm_valid = valid_prep.filter(F.col("carga") > 0)

    if WINSORIZE_LABELS:
        gm_train, q_hi = winsorize_label(gm_train, WINSOR_Q, "carga")
        print(f"[Gamma] Winsor TRAIN p{int(WINSOR_Q*1000)/10}% = {q_hi:.6f}")
    gm_train = enforce_gamma_min(gm_train, "carga", GAMMA_MIN_POS)

    scale = compute_label_scale(gm_train, label="carga", q=0.99)
    gm_train = gm_train.withColumn("carga_gm_s_raw", F.col("carga")/F.lit(scale))
    gm_valid = gm_valid.withColumn("carga_gm_s_raw", F.col("carga")/F.lit(scale))
    gm_train = gm_train.withColumn("carga_gm_s", F.when(F.col("carga_gm_s_raw") < F.lit(GLM_SCALED_Y_MIN),
                                                       F.lit(GLM_SCALED_Y_MIN)).otherwise(F.col("carga_gm_s_raw")))
    gm_valid = gm_valid.withColumn("carga_gm_s", F.when(F.col("carga_gm_s_raw") < F.lit(GLM_SCALED_Y_MIN),
                                                       F.lit(GLM_SCALED_Y_MIN)).otherwise(F.col("carga_gm_s_raw")))

    if GLM_USE_OFFSET:
        mu0 = float(gm_train.agg(F.avg("carga_gm_s")).first()[0] or 1.0)
        off_val = math.log(max(mu0, GLM_SCALED_Y_MIN))
        gm_train = ensure_constant_offset(gm_train, "offset_gm", off_val)
        gm_valid = ensure_constant_offset(gm_valid, "offset_gm", off_val)

    is_finite = vector_is_finite_udf()
    has_nnz   = vector_has_nnz_udf()
    gm_train = gm_train.filter(is_finite("features_glm")).filter(has_nnz("features_glm"))
    gm_valid = gm_valid.filter(is_finite("features_glm")).filter(has_nnz("features_glm"))

    n_pos = gm_train.count()
    if n_pos == 0:
        raise RuntimeError("No hay positivos en TRAIN para Gamma tras filtros.")

    fit_cols = ["features_glm", "carga_gm_s"] + (["offset_gm"] if GLM_USE_OFFSET else [])
    gm_fit = gm_train.select(*fit_cols)

    evaluator_mae = RegressionEvaluator(labelCol="carga", predictionCol="pred_carga", metricName="mae")

    # 1) Gamma log
    try:
        gm1 = GeneralizedLinearRegression(
            featuresCol="features_glm", labelCol="carga_gm_s",
            family="gamma", link="log",
            predictionCol="pred_gm_s",
            maxIter=200, regParam=1e-2, tol=1e-6,
            offsetCol=("offset_gm" if GLM_USE_OFFSET else None)
        ).fit(gm_fit)
        p1 = gm1.transform(gm_valid).withColumn("pred_carga", F.col("pred_gm_s")*F.lit(scale))
        p1 = cap_predictions_by_label_quantile(gm_train, p1, label="carga", pred_col="pred_carga", q=0.99, factor=1.0)
        _ = evaluator_mae.evaluate(p1)
        return gm1, p1, gm_valid, {"scale": scale, "type": "gamma_log", "offset": (off_val if GLM_USE_OFFSET else None)}
    except Exception as e1:
        print(f"[AVISO] Gamma(link=log) falló: {e1}")

    # 2) Gamma inverse
    try:
        gm2 = GeneralizedLinearRegression(
            featuresCol="features_glm", labelCol="carga_gm_s",
            family="gamma", link="inverse",
            predictionCol="pred_gm_s",
            maxIter=200, regParam=1e-1, tol=1e-6,
            offsetCol=("offset_gm" if GLM_USE_OFFSET else None)
        ).fit(gm_fit)
        p2 = gm2.transform(gm_valid).withColumn("pred_carga", F.col("pred_gm_s")*F.lit(scale))
        p2 = cap_predictions_by_label_quantile(gm_train, p2, label="carga", pred_col="pred_carga", q=0.99, factor=1.0)
        _ = evaluator_mae.evaluate(p2)
        return gm2, p2, gm_valid, {"scale": scale, "type": "gamma_inv", "offset": (off_val if GLM_USE_OFFSET else None)}
    except Exception as e2:
        print(f"[AVISO] Gamma(link=inverse) falló: {e2}")

    # 3) Fallback: Lognormal (GLM gaussian sobre log(y_s)) SIN offset
    gm_train_ln = gm_train.withColumn("log_carga_s", F.log(F.col("carga_gm_s")))
    gm_valid_ln = gm_valid.withColumn("log_carga_s", F.log(F.col("carga_gm_s")))
    try:
        ln = GeneralizedLinearRegression(
            featuresCol="features_glm", labelCol="log_carga_s",
            family="gaussian", link="identity",
            predictionCol="pred_log_s",
            maxIter=200, regParam=5e-3, tol=1e-6
        ).fit(gm_train_ln.select("features_glm","log_carga_s"))
        p_ln = ln.transform(gm_valid_ln).withColumn("pred_carga", F.exp(F.col("pred_log_s"))*F.lit(scale))
        p_ln = cap_predictions_by_label_quantile(gm_train, p_ln, label="carga", pred_col="pred_carga", q=0.99, factor=1.0)
        return ln, p_ln, gm_valid, {"scale": scale, "type": "lognormal", "offset": None}
    except Exception as e3:
        print(f"[ERROR] Fallback lognormal también falló: {e3}")
        raise

# =====================
# Two-part con severidad GBT en positivos
# =====================

def train_severity_gbt_pos(train_prep, valid_prep, base_params):
    trp = (train_prep
           .filter(F.col("carga") > 0)
           .withColumn("log1p_carga_pos", F.log1p(F.col("carga"))))
    vap = valid_prep.filter(F.col("carga") > 0)

    gbt = GBTRegressor(
        featuresCol="features_raw",
        labelCol="log1p_carga_pos",
        seed=42,
        maxDepth=base_params.get("maxDepth", 6),
        maxIter=base_params.get("maxIter", 100),
        stepSize=base_params.get("stepSize", 0.1)
    )
    sev_model = gbt.fit(trp)

    log_var = compute_log_residual_var(
        sev_model, trp, label_col="log1p_carga_pos", pred_col="prediction"
    )

    sev_valid = (sev_model.transform(vap)
                 .withColumn("sev_pred", F.exp(F.col("prediction") + F.lit(0.5*log_var)) - F.lit(1.0)))

    sev_valid = cap_predictions_by_label_quantile(
        train_prep, sev_valid, label="carga", pred_col="sev_pred", q=0.99, factor=1.0
    )

    return sev_model, sev_valid.select("siniestro_hash", "sev_pred")

def train_two_part(train_prep, valid_prep, base_params):
    # Frecuencia
    tr = train_prep.withColumn("y_pos", (F.col("carga") > 0).cast("int"))
    lr = LogisticRegression(featuresCol="features_glm", labelCol="y_pos",
                            maxIter=100, regParam=1e-3, elasticNetParam=0.0,
                            predictionCol="pred_cls", probabilityCol="prob_cls", rawPredictionCol="raw_cls")
    lr_m = lr.fit(tr)

    pv_valid = lr_m.transform(valid_prep)
    get_p1 = F.udf(lambda v: float(v[1]), "double")
    pv_valid = pv_valid.withColumn("p_pos", get_p1(F.col("prob_cls")))

    # Severidad con GBT positivos
    sev_model, sev_valid = train_severity_gbt_pos(train_prep, valid_prep, base_params)
    pv_valid = (pv_valid
                .join(sev_valid, on="siniestro_hash", how="left")
                .fillna({"sev_pred": 0.0}))

    final = pv_valid.withColumn("pred_carga", F.col("p_pos") * F.col("sev_pred"))
    final = cap_predictions_by_label_quantile(
        train_prep, final, label="carga", pred_col="pred_carga", q=0.99, factor=1.0
    )
    return lr_m, sev_model, final

# =====================
# Baseline / Blend / per-LOB
# =====================

def baseline_per_lob(train_lob, valid_lob):
    return baseline_mean(train_lob, valid_lob, ["segmento_cliente_detalle"])

def train_gbt_per_lob(df_global_raw, lob_value, base_params, winsor_train=True, winsor_q=0.999,
                      key_col="siniestro_hash", prefer_metric="rmse"):
    """
    prefer_metric: 'rmse' o 'mae'
    """
    df_lob = df_global_raw.filter(F.col("lob") == F.lit(lob_value))
    df_lob, num_cols_lob = feature_engineering(df_lob)
    tr, va = temporal_split(df_lob, ANIO_VALID)

    te_cols = []
    if "codigo_postal_norm" in df_lob.columns:
        tr, va = add_target_encoding(tr, va, ["codigo_postal_norm"], name="te_cp")
        te_cols.append("te_cp_mean")
    tr, va = add_target_encoding(tr, va, ["segmento_cliente_detalle"], name="te_seg")
    te_cols.append("te_seg_mean")

    if winsor_train:
        try:
            tr, q_hi_lob = winsorize_label(tr, winsor_q, "carga")
            print(f"[LOB={lob_value}] Winsor TRAIN p{int(winsor_q*1000)/10}% = {q_hi_lob:.2f}")
            tr = tr.withColumn("log_carga", F.log1p(F.col("carga")))
        except Exception:
            pass

    prep_pipeline_lob = build_prep_pipeline(num_cols_lob, ["segmento_cliente_detalle"],
                                            glm_extra_cols=[c for c in te_cols if c in tr.columns])
    _, tr_p, va_p = prepare_datasets(tr, va, prep_pipeline_lob)

    # Modelo per-LOB (misma config que global)
    best_gbt_lob = GBTRegressor(
        featuresCol="features_raw", labelCol="log_carga", seed=42,
        maxDepth=base_params.get("maxDepth", 6),
        maxIter=base_params.get("maxIter", 100),
        stepSize=base_params.get("stepSize", 0.1)
    ).fit(tr_p)

    log_var = compute_log_residual_var(best_gbt_lob, tr_p, label_col="log_carga", pred_col="prediction")
    pred_lob = (best_gbt_lob.transform(va_p)
                .withColumn("pred_carga", F.exp(F.col("prediction") + F.lit(0.5*log_var)) - F.lit(1.0)))

    evaluator_rmse = RegressionEvaluator(labelCol="carga", predictionCol="pred_carga", metricName="rmse")
    evaluator_mae  = RegressionEvaluator(labelCol="carga", predictionCol="pred_carga", metricName="mae")
    rmse_m = evaluator_rmse.evaluate(pred_lob); mae_m = evaluator_mae.evaluate(pred_lob)
    print(f"[LOB={lob_value}] GBT per-LOB → RMSE={rmse_m:,.2f} | MAE={mae_m:,.2f} | {extra_metrics_smape(pred_lob)}")

    base_lob = baseline_per_lob(tr, va)
    rmse_b = evaluator_rmse.evaluate(base_lob); mae_b = evaluator_mae.evaluate(base_lob)
    print(f"[LOB={lob_value}] Baseline (seg) → RMSE={rmse_b:,.2f} | MAE={mae_b:,.2f} | {extra_metrics_smape(base_lob)}")

    lam_lob, blended = tune_blend_lambda(pred_lob, base_lob, metric=prefer_metric, key_col=key_col)
    rmse_bl = evaluator_rmse.evaluate(blended); mae_bl = evaluator_mae.evaluate(blended)
    print(f"[LOB={lob_value}] BLEND (λ*={lam_lob:.2f}) → RMSE={rmse_bl:,.2f} | MAE={mae_bl:,.2f} | {extra_metrics_smape(blended)}")

    # Selección de ganador robusta
    # prioridad en empate: blend > baseline > model
    prio = {"blend":0, "baseline":1, "model":2}
    cand = [
        ("model",    rmse_m, mae_m),
        ("baseline", rmse_b, mae_b),
        (f"blend(λ={lam_lob:.2f})", rmse_bl, mae_bl)
    ]
    if prefer_metric == "mae":
        winner, rmse_best, mae_best = sorted(
            cand, key=lambda x: (x[2], prio[x[0].split('(')[0]])
        )[0]
    else:
        winner, rmse_best, mae_best = sorted(
            cand, key=lambda x: (x[1], prio[x[0].split('(')[0]])
        )[0]
    print(f"[LOB={lob_value}] Mejor en VALID: {winner}")
    return rmse_best, mae_best

# =====================
# Main
# =====================

def main():
    spark = build_spark()

    # 1) Carga cruda y FE global
    df_raw = load_aggregated_delta(spark)
    df, num_cols = feature_engineering(df_raw)
    cat_cols = ["lob", "segmento_cliente_detalle"]

    # 2) Split
    train, valid = temporal_split(df, ANIO_VALID)
    n_train, n_valid = train.count(), valid.count()
    print(f"Train: {n_train:,} | Valid: {n_valid:,} (Año valid = {ANIO_VALID})")
    if n_train == 0 or n_valid == 0:
        print("Advertencia: split vacío. Revisa ANIO_VALID/histórico.")
        sys.exit(1)

    # 3) Auditoría
    quick_label_audit(train, "TRAIN raw")
    quick_label_audit(valid, "VALID raw")

    # 4) Target encodings (globales) → para GLM features_glm
    glm_extra = []
    if "codigo_postal_norm" in train.columns:
        train, valid = add_target_encoding(train, valid, ["codigo_postal_norm"], name="te_cp")
        glm_extra.append("te_cp_mean")
    train, valid = add_target_encoding(train, valid, ["lob","segmento_cliente_detalle"], name="te_lob_seg")
    glm_extra.append("te_lob_seg_mean")

    # 5) Preparación (fit en TRAIN): GLM usa numéricas + glm_extra; GBT usa numéricas + OHE
    prep_pipeline = build_prep_pipeline(num_cols, cat_cols, glm_extra_cols=[c for c in glm_extra if c in train.columns])
    prep_model, train_prep, valid_prep = prepare_datasets(train, valid, prep_pipeline)

    # 6) Evaluadores y recolector
    evaluator_rmse = RegressionEvaluator(labelCol="carga", predictionCol="pred_carga", metricName="rmse")
    evaluator_mae  = RegressionEvaluator(labelCol="carga", predictionCol="pred_carga", metricName="mae")
    results = init_results()

    # 7) Modelos globales
    # 7.1 GBT GLOBAL (tuning)
    best_gbt, pred_gbt, best_params = train_gbt_log(train_prep, valid_prep, tune=True, parallelism=1)
    add_result(results, "GBT (tuned+bias)", pred_gbt)

    # 7.2 Tweedie
    try:
        model_tw, pred_tw, best_v = tune_tweedie(train_prep, valid_prep, TWEEDIE_VGRID)
        add_result(results, f"Tweedie (v={best_v})", pred_tw, extra_fn=lambda: {"variancePower": float(best_v)})
    except Exception as e:
        print(f"[AVISO] Tweedie falló: {e}")
        model_tw, pred_tw, best_v = None, None, None

    # 7.3 Gamma (>0) (opcional)
    if USE_GAMMA_MODELS:
        try:
            model_gm, pred_gm, _, info_gm = train_glm_gamma(train_prep, valid_prep)
            add_result(results, f"Gamma (>0) robusto [{info_gm['type']}]", pred_gm)
        except Exception as e:
            print(f"[AVISO] Gamma falló: {e}")

    # 7.4 Two-part (LR × Severidad GBT)
    if USE_TWO_PART:
        try:
            lr_pos, sev_gbt_pos, pred_two = train_two_part(train_prep, valid_prep, best_params)
            add_result(results, "Two-part (LR·SevGBT)", pred_two)
            # Guardado modelos two-part
            lr_pos.write().overwrite().save(str(MODEL_PATH_LR_POS))
            sev_gbt_pos.write().overwrite().save(str(MODEL_PATH_SEV_GBT_POS))
        except Exception as e:
            print(f"[AVISO] Two-part falló: {e}")
            pred_two = None

    # 8) Calibración por deciles (CSV)
    cal_gbt = decile_calibration(pred_gbt.repartition(DECILES_REPARTITION), "carga", "pred_carga")
    (cal_gbt.coalesce(1).write.mode("overwrite").option("header", True)
        .csv(str(DIAG_DIR / "calibration_deciles_gbt_valid")))
    if pred_tw is not None:
        cal_tw = decile_calibration(pred_tw.repartition(DECILES_REPARTITION), "carga", "pred_carga")
        (cal_tw.coalesce(1).write.mode("overwrite").option("header", True)
            .csv(str(DIAG_DIR / "calibration_deciles_tweedie_valid")))
    if USE_GAMMA_MODELS and 'pred_gm' in locals() and pred_gm is not None:
        cal_gm = decile_calibration(pred_gm.repartition(DECILES_REPARTITION), "carga", "pred_carga")
        (cal_gm.coalesce(1).write.mode("overwrite").option("header", True)
            .csv(str(DIAG_DIR / "calibration_deciles_gamma_valid_pos")))
    if USE_TWO_PART and pred_two is not None:
        cal_two = decile_calibration(pred_two.repartition(DECILES_REPARTITION), "carga", "pred_carga")
        (cal_two.coalesce(1).write.mode("overwrite").option("header", True)
            .csv(str(DIAG_DIR / "calibration_deciles_two_part_valid")))

    # 9) Baselines globales
    base1 = baseline_mean(train, valid, ["lob"])
    base2 = baseline_mean(train, valid, ["lob","segmento_cliente_detalle"])
    add_result(results, "Baseline: lob", base1)
    add_result(results, "Baseline: lob×segmento", base2)

    # 10) Blends globales: RMSE y MAE
    try:
        lam_star_rmse, blended_global_rmse = tune_blend_lambda(pred_gbt, base2, metric="rmse", key_col="siniestro_hash")
        add_result(results, f"Blend GBT↔lob×seg (λ*={lam_star_rmse:.2f}, RMSE)", blended_global_rmse)

        lam_star_mae, blended_global_mae = tune_blend_lambda(pred_gbt, base2, metric="mae", key_col="siniestro_hash")
        add_result(results, f"Blend GBT↔lob×seg (λ*={lam_star_mae:.2f}, MAE)", blended_global_mae)
    except Exception as e:
        print(f"[AVISO] Blend global falló: {e}")

    # 11) Blend por deciles (RMSE y MAE)
    try:
        blended_decile_rmse, lam_map_rmse = tune_blend_by_decile(pred_gbt, base2, metric="rmse",
                                                                  key_col="siniestro_hash", num_deciles=10)
        add_result(results, "Blend por deciles (RMSE)", blended_decile_rmse)
        print(f"[INFO] λ por decil (RMSE): {lam_map_rmse}")
    except Exception as e:
        print(f"[AVISO] Blend por deciles (RMSE) falló: {e}")

    try:
        blended_decile_mae, lam_map_mae = tune_blend_by_decile(pred_gbt, base2, metric="mae",
                                                                key_col="siniestro_hash", num_deciles=10)
        add_result(results, "Blend por deciles (MAE)", blended_decile_mae)
        print(f"[INFO] λ por decil (MAE): {lam_map_mae}")
    except Exception as e:
        print(f"[AVISO] Blend por deciles (MAE) falló: {e}")

    # 12) Fallback top 1% (Two-part y GBT)
    try:
        if USE_TWO_PART and 'pred_two' in locals() and pred_two is not None:
            pred_two_fb, thr_two = apply_top_percentile_fallback(pred_two, base2, key_col="siniestro_hash", top_p=0.01)
            add_result(results, f"Two-part + fallback top1% (thr={thr_two:,.0f})", pred_two_fb)

        pred_gbt_fb, thr_gbt = apply_top_percentile_fallback(pred_gbt, base2, key_col="siniestro_hash", top_p=0.01)
        add_result(results, f"GBT + fallback top1% (thr={thr_gbt:,.0f})", pred_gbt_fb)
    except Exception as e:
        print(f"[AVISO] Fallback top1% falló: {e}")

    # 13) Guardado modelos globales
    prep_model.write().overwrite().save(str(MODEL_PATH_PREP))
    best_gbt.write().overwrite().save(str(MODEL_PATH_GBT))
    if 'model_tw' in locals() and model_tw is not None:
        model_tw.write().overwrite().save(str(MODEL_PATH_TW))
    if USE_GAMMA_MODELS and 'model_gm' in locals() and model_gm is not None:
        model_gm.write().overwrite().save(str(MODEL_PATH_GAMMA))

    # 14) Resumen + CSV
    flush_results(results, spark, DIAG_DIR)

    # 15) Per-LOB (reuso hiperparámetros)
    if USE_PER_LOB:
        lob_counts = (valid.groupBy("lob").count().orderBy(F.desc("count"))).collect()
        cand_lobs = [r["lob"] for r in lob_counts if r["count"] >= 2000]
        print(f"\nProbaré modelos per-LOB en: {cand_lobs}")
        for lobv in cand_lobs:
            try:
                train_gbt_per_lob(df_raw, lobv, best_params, winsor_train=True, winsor_q=0.999,
                                  key_col="siniestro_hash", prefer_metric="rmse")
            except Exception as e:
                print(f"[AVISO] per-LOB {lobv} falló: {e}")

    # =====================================================================
    # === SCORING / EXPORT DUAL (campeón MAE y RMSE) + MEMORIA + PLOTS  ===
    # =====================================================================
    from math import sqrt

    # Carpeta de diagnósticos
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- Helpers ----------
    def _choose_champion(results_list, preds_store, prefer_metric="mae"):
        """Elige campeón según métrica (mae|rmse) con prioridad: blends > two-part > gbt > tweedie > baseline."""
        prio = {
            "Blend por deciles": 0, "Blend GBT↔lob×seg": 0, "Blend": 0,
            "fallback": 1,
            "Two-part": 2,
            "GBT (tuned+bias)": 3,
            "Tweedie": 4,
            "Baseline": 5
        }
        cand = [r for r in results_list if r["model"] in preds_store]
        if not cand:
            raise RuntimeError("No hay candidatos con predicciones disponibles.")

        def _key(r):
            main = r["MAE"] if prefer_metric == "mae" else r["RMSE"]
            tag = next((k for k in prio if k in r["model"]), "Baseline")
            return (main, prio[tag])

        best = sorted(cand, key=_key)[0]
        return best["model"], preds_store[best["model"]]

    def _export_predictions(champion_name, champion_df, tag_suffix):
        """
        Guarda:
          - pred por siniestro (Delta+CSV) con columnas: codigo_postal_norm, segmento_cliente_detalle, lob, pred_euros
          - agregada por cp×seg×lob (Delta+CSV)
          - distribución de errores (CSV)
          - curva por deciles (CSV)
          - métricas por LOB (CSV)
        Devuelve: (tabla_pred, tabla_agg, base_rows, base_agg)
        """
        # Contexto + etiqueta desde VALID
        ctx_all = (valid
                   .select("siniestro_hash", "codigo_postal_norm", "segmento_cliente_detalle", "lob", "carga")
                   .dropDuplicates(["siniestro_hash"]))

        # Pred completo con contexto
        full = (champion_df
                .select("siniestro_hash", F.col("pred_carga").alias("pred_euros"))
                .join(ctx_all, on="siniestro_hash", how="left"))

        # Tabla por siniestro (solo columnas pedidas)
        tabla_pred = full.select("codigo_postal_norm", "segmento_cliente_detalle", "lob", "pred_euros")

        # Tabla agregada
        tabla_agg = (full.groupBy("codigo_postal_norm", "segmento_cliente_detalle", "lob")
                     .agg(F.sum("pred_euros").alias("pred_total_eur"),
                          F.avg("pred_euros").alias("pred_medio_eur"),
                          F.count("*").alias("n_siniestros"))
                     .orderBy("codigo_postal_norm", "segmento_cliente_detalle", "lob"))

        # Rutas de salida
        PRED_DIR = REPO_ROOT / "data" / "predictions"
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        base_rows = str(PRED_DIR / f"pred_siniestro_valid_{ANIO_VALID}_{tag_suffix}_{ts}")
        base_agg = str(PRED_DIR / f"pred_agg_cp_seg_lob_valid_{ANIO_VALID}_{tag_suffix}_{ts}")

        # Guardado Delta
        tabla_pred.write.mode("overwrite").format("delta").save(base_rows + "_delta")
        tabla_agg.write.mode("overwrite").format("delta").save(base_agg + "_delta")
        # Guardado CSV
        (tabla_pred.coalesce(1).write.mode("overwrite").option("header", True).csv(base_rows + "_csv"))
        (tabla_agg.coalesce(1).write.mode("overwrite").option("header", True).csv(base_agg + "_csv"))

        print(f"[SCORING/{tag_suffix}] Campeón: {champion_name}")
        print(f"[SCORING/{tag_suffix}] Guardado filas: {base_rows}_delta  /  {base_rows}_csv")
        print(f"[SCORING/{tag_suffix}] Guardado agregada: {base_agg}_delta  /  {base_agg}_csv")

        # ===== Artefactos de memoria =====
        DIAG_THIS = DIAG_DIR / f"memoria_{tag_suffix}_{ts}"
        DIAG_THIS.mkdir(parents=True, exist_ok=True)

        # 1) Distribución de errores (cuantiles)
        err = (full
               .withColumn("error", F.col("pred_euros") - F.col("carga"))
               .withColumn("abs_error", F.abs(F.col("error"))))
        probs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        q_err = err.approxQuantile("error", probs, 0.001)
        q_abs = err.approxQuantile("abs_error", probs, 0.001)
        rows = []
        for i, p in enumerate(probs):
            rows.append(("error", p, float(q_err[i])))
            rows.append(("abs_err", p, float(q_abs[i])))
        dist_df = spark.createDataFrame(rows, ["metric", "quantile", "value"])
        (dist_df.coalesce(1).write.mode("overwrite").option("header", True)
         .csv(str(DIAG_THIS / "error_distribution_csv")))
        print(f"[MEMORIA/{tag_suffix}] Distribución de errores → {DIAG_THIS / 'error_distribution_csv'}")

        # 2) Curva por deciles (observado vs predicho)
        dec = decile_calibration(full, label="carga", pred="pred_euros")
        (dec.coalesce(1).write.mode("overwrite").option("header", True)
         .csv(str(DIAG_THIS / "decile_curve_csv")))
        print(f"[MEMORIA/{tag_suffix}] Curva por deciles → {DIAG_THIS / 'decile_curve_csv'}")

        # 3) Métricas por LOB (MAE, RMSE)
        bylob = (full.groupBy("lob")
                 .agg(
            F.sqrt(F.avg((F.col("pred_euros") - F.col("carga")) ** 2)).alias("RMSE"),
            F.avg(F.abs(F.col("pred_euros") - F.col("carga"))).alias("MAE"),
            F.count("*").alias("n")
        )
                 .orderBy("lob"))
        (bylob.coalesce(1).write.mode("overwrite").option("header", True)
         .csv(str(DIAG_THIS / "metrics_by_lob_csv")))
        print(f"[MEMORIA/{tag_suffix}] MAE/RMSE por LOB → {DIAG_THIS / 'metrics_by_lob_csv'}")

        return tabla_pred, tabla_agg, base_rows, base_agg

    # --- 1) Almacén de predicciones con TODO lo disponible ---
    preds_store = {}
    preds_store["GBT (tuned+bias)"] = pred_gbt
    if 'pred_tw' in locals() and pred_tw is not None:
        preds_store[f"Tweedie (v={best_v})"] = pred_tw
    if 'pred_two' in locals() and pred_two is not None:
        preds_store["Two-part (LR·SevGBT)"] = pred_two
    preds_store["Baseline: lob"] = base1
    preds_store["Baseline: lob×segmento"] = base2
    if 'blended_global_rmse' in locals():
        preds_store[f"Blend GBT↔lob×seg (λ*={lam_star_rmse:.2f}, RMSE)"] = blended_global_rmse
    if 'blended_global_mae' in locals():
        preds_store[f"Blend GBT↔lob×seg (λ*={lam_star_mae:.2f}, MAE)"] = blended_global_mae
    if 'blended_decile_rmse' in locals():
        preds_store["Blend por deciles (RMSE)"] = blended_decile_rmse
    if 'blended_decile_mae' in locals():
        preds_store["Blend por deciles (MAE)"] = blended_decile_mae
    if 'pred_two_fb' in locals():
        preds_store[f"Two-part + fallback top1% (thr={thr_two:,.0f})"] = pred_two_fb
    if 'pred_gbt_fb' in locals():
        preds_store[f"GBT + fallback top1% (thr={thr_gbt:,.0f})"] = pred_gbt_fb

    # --- 2) Elegir CAMPEONES por MAE y por RMSE ---
    champ_mae_name, champ_mae_df = _choose_champion(results, preds_store, prefer_metric="mae")
    champ_rmse_name, champ_rmse_df = _choose_champion(results, preds_store, prefer_metric="rmse")
    print(f"\n[SCORING] Campeón MAE:  {champ_mae_name}")
    print(f"[SCORING] Campeón RMSE: {champ_rmse_name}")

    # --- 3) Exportar ambas tablas + artefactos de memoria ---
    tabla_pred_mae, tabla_agg_mae, out_rows_mae, out_agg_mae = _export_predictions(champ_mae_name, champ_mae_df,
                                                                                   "champ_mae")
    tabla_pred_rmse, tabla_agg_rmse, out_rows_rmse, out_agg_rmse = _export_predictions(champ_rmse_name, champ_rmse_df,
                                                                                       "champ_rmse")

    # --- 4) Muestras en consola ---
    print("\n[SCORING] Muestra (siniestro / MAE):")
    tabla_pred_mae.show(10, truncate=False)
    print("\n[SCORING] Muestra (agregada CP×seg×LOB / MAE):")
    tabla_agg_mae.show(10, truncate=False)

    print("\n[SCORING] Muestra (siniestro / RMSE):")
    tabla_pred_rmse.show(10, truncate=False)
    print("\n[SCORING] Muestra (agregada CP×seg×LOB / RMSE):")
    tabla_agg_rmse.show(10, truncate=False)

    # ==========================================================
    # === PLOTS: Distribución de errores para TODOS los modelos
    # ==========================================================
    from functools import reduce
    try:
        import matplotlib
        matplotlib.use("Agg")  # backend no interactivo (servidor/cluster)
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        MATP_OK = True
    except Exception as _e:
        print("[WARN] matplotlib/pandas no disponible. Solo exportaré CSV de errores capados.")
        MATP_OK = False

    # Construimos DF de errores de TODOS los modelos
    ctx_err = valid.select("siniestro_hash", "carga").dropDuplicates(["siniestro_hash"])
    err_dfs = []
    for name, dfm in preds_store.items():
        if dfm is None:
            continue
        cols = dfm.columns
        if "siniestro_hash" not in cols or "pred_carga" not in cols:
            continue
        df_err = (
            dfm.select("siniestro_hash", F.col("pred_carga").alias("pred"))
            .join(ctx_err, on="siniestro_hash", how="inner")
            .select(
                F.lit(name).alias("model"),
                (F.col("pred") - F.col("carga")).alias("error")
            )
            .withColumn("ae", F.abs(F.col("error")))
        )
        err_dfs.append(df_err.select("model", "error", "ae"))

    if not err_dfs:
        print("[PLOTS] No hay DFs de predicciones con 'pred_carga'. Omito gráficos.")
    else:
        errors_all = reduce(lambda a, b: a.unionByName(b), err_dfs)

        # p99 global de |error| para capar en las gráficas
        p99 = float(errors_all.approxQuantile("ae", [0.99], 0.001)[0])
        errors_all_cap = errors_all.withColumn("ae_cap", F.when(F.col("ae") > p99, p99).otherwise(F.col("ae")))

        # Pasamos a pandas
        pdf = errors_all_cap.toPandas()

        # Carpeta de salida
        ts_plots = datetime.now().strftime("%Y%m%d_%H%M%S")
        PLOTS_DIR = DIAG_DIR / f"plots_all_models_{ANIO_VALID}_{ts_plots}"
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)

        # Guardamos CSV crudo capado
        csv_path = PLOTS_DIR / "errors_all_models_cap.csv"
        pdf.to_csv(csv_path, index=False)
        with open(PLOTS_DIR / "README.txt", "w") as f:
            f.write(f"Cap visual en p99(|error|) = {p99:.4f} euros\n")
            f.write(f"Modelos: {sorted(pdf['model'].unique().tolist())}\n")

        if not MATP_OK:
            print("[PLOTS] matplotlib no disponible; dejé CSV en:", csv_path)
        else:
            models = sorted(pdf["model"].unique().tolist())

            # (A) ECDF de |error| capado
            import numpy as np
            import matplotlib.pyplot as plt
            plt.figure(figsize=(9, 6))
            for m in models:
                v = np.sort(pdf.loc[pdf.model == m, "ae_cap"].values.astype(float))
                if v.size == 0:
                    continue
                y = np.arange(1, v.size + 1) / v.size
                plt.plot(v, y, label=m)
            plt.xlabel(f"|error| (euros)  [cap p99={p99:,.0f}]")
            plt.ylabel("ECDF")
            plt.title("Distribución acumulada del error absoluto por modelo")
            plt.grid(True, alpha=0.3)
            plt.legend(loc="lower right", fontsize=8)
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / "A_ecdf_abs_error.png", dpi=150)
            plt.close()

            # (B) Histograma (densidad) de |error| capado
            plt.figure(figsize=(9, 6))
            bins = 50
            for m in models:
                d = pdf.loc[pdf.model == m, "ae_cap"].values.astype(float)
                if d.size == 0:
                    continue
                plt.hist(d, bins=bins, histtype="step", density=True, label=m)
            plt.xlabel(f"|error| (euros)  [cap p99={p99:,.0f}]")
            plt.ylabel("Densidad")
            plt.title("Histograma (densidad) de |error| por modelo")
            plt.grid(True, alpha=0.3)
            plt.legend(loc="upper right", fontsize=8)
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / "B_hist_abs_error.png", dpi=150)
            plt.close()

            # (C) Boxplot de |error| capado por modelo (ordenado por mediana)
            med = pdf.groupby("model")["ae_cap"].median().sort_values()
            order = med.index.tolist()
            data = [pdf.loc[pdf.model == m, "ae_cap"].values.astype(float) for m in order]
            plt.figure(figsize=(max(10, 0.45 * len(order) + 6), 6))
            plt.boxplot(data, labels=order, showfliers=False)
            plt.ylabel(f"|error| (euros)  [cap p99={p99:,.0f}]")
            plt.title("Boxplot de |error| por modelo (sin outliers)")
            plt.xticks(rotation=30, ha="right", fontsize=8)
            plt.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / "C_boxplot_abs_error.png", dpi=150)
            plt.close()

            # (D) Curva de concentración (Lorenz/Pareto) del error absoluto
            plt.figure(figsize=(9, 6))
            for m in models:
                d = pdf.loc[pdf.model == m, "ae"].values.astype(float)
                if d.size == 0 or np.nansum(d) == 0:
                    continue
                v = np.sort(d)  # ascendente
                x = np.arange(1, v.size + 1) / v.size
                cum = np.cumsum(v) / v.sum()
                plt.plot(x, cum, label=m)
            plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)  # diagonal
            plt.xlabel("Proporción de siniestros")
            plt.ylabel("Proporción acumulada de |error|")
            plt.title("Concentración del error absoluto (Lorenz/Pareto)")
            plt.grid(True, alpha=0.3)
            plt.legend(loc="lower right", fontsize=8)
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / "D_lorenz_abs_error.png", dpi=150)
            plt.close()

            print(f"[PLOTS] Gráficos guardados en: {PLOTS_DIR}")

    spark.stop()

if __name__ == "__main__":
    main()


