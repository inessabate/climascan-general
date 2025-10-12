#!/usr/bin/env python3
"""
Train Logistic Regression to predict `ocurre_siniestro` using meteorological features.

- Read from Delta (table/path) or Parquet
- Temporal split by year (e.g., train=2021, test=2022)
- Impute numeric features with mean
- Optional standardization
- Class imbalance via class weights (weightCol)
- Cross-validated LR (optimize AUC-ROC)
- Export TEST predictions aggregated at ZIP level:
    codigo_postal, prob_ocurre_siniestro  (mean prob across test dates)

Project-aware defaults for your tree:
- Recommended input path (Delta): data/aggregated/classification_deltalake
- Default output dir: src/data_analytics/models/classification/logreg/artifacts

Example
-------
spark-submit \
  --packages io.delta:delta-spark_2.12:3.2.0 \
  src/data_analytics/models/classification/logreg/train_logreg.py \
  --input "data/aggregated/classification_deltalake" \
  --input-format delta \
  --date-col fecha \
  --zip-col codigo_postal \
  --train-years 2021 \
  --test-years 2022 \
  --label ocurre_siniestro \
  --features "tmed,tmin,tmax,prec,hrmedia,hrmax,hrmin,velmedia,racha" \
  --standardize \
  --output-dir src/data_analytics/models/classification/logreg/artifacts
"""

import argparse
import json
import os
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.mllib.evaluation import MulticlassMetrics


def parse_args():
    p = argparse.ArgumentParser(description="Train Logistic Regression baseline")
    p.add_argument("--input", required=True, help="Delta table name or Parquet/Delta path")
    p.add_argument("--input-format", choices=["delta", "parquet"], default="delta")
    p.add_argument("--date-col", default="fecha", help="Date column (string or date)")
    p.add_argument("--zip-col", default="codigo_postal", help="ZIP/postal code column")
    p.add_argument("--train-years", required=True, help="Comma-separated train years (e.g., 2021 or 2020,2021)")
    p.add_argument("--test-years", required=True, help="Comma-separated test years (e.g., 2022)")
    p.add_argument("--label", default="ocurre_siniestro", help="Binary label column 0/1")
    p.add_argument(
        "--features",
        default="tmed,tmin,tmax,prec,hrmedia,hrmax,hrmin,velmedia,racha",
        help="Comma-separated numeric feature columns",
    )
    p.add_argument("--standardize", action="store_true", help="Standardize numeric features")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="src/data_analytics/models/classification/logreg/artifacts")
    return p.parse_args()


def build_spark():
    return (
        SparkSession.builder
        .appName("tfm-train-logreg")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "200")
        # Delta configs necesarias:
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )



def read_data(spark: SparkSession, args):
    if args.input_format == "delta":
        if os.path.isdir(args.input):
            df = spark.read.format("delta").load(args.input)
        else:
            df = spark.read.table(args.input)
    else:
        df = spark.read.parquet(args.input)
    return df


def ensure_binary_numeric(df, label_col):
    if dict(df.dtypes)[label_col] not in ("double", "float", "int", "bigint", "smallint", "tinyint", "decimal"):
        df = df.withColumn(label_col, F.col(label_col).cast("double"))
    return df


def temporal_split(df, date_col, train_years, test_years):
    df = df.withColumn("_date", F.to_date(F.col(date_col)))
    df = df.withColumn("_year", F.year("_date"))
    train_df = df.filter(F.col("_year").isin(train_years))
    test_df = df.filter(F.col("_year").isin(test_years))
    return train_df, test_df


def build_pipeline(feature_cols, standardize=False, label_col="ocurre_siniestro"):
    stages = []
    # Impute numeric
    imputer = Imputer(inputCols=feature_cols, outputCols=[f"{c}__imp" for c in feature_cols])
    stages.append(imputer)

    # Assemble → optional scale
    assembler = VectorAssembler(inputCols=[f"{c}__imp" for c in feature_cols], outputCol="num_vec")
    stages.append(assembler)

    if standardize:
        scaler = StandardScaler(inputCol="num_vec", outputCol="features", withStd=True, withMean=False)
        stages.append(scaler)
    else:
        stages.append(VectorAssembler(inputCols=["num_vec"], outputCol="features"))

    lr = LogisticRegression(
        featuresCol="features",
        labelCol=label_col,
        probabilityCol="probability",
        rawPredictionCol="rawPrediction",
        maxIter=100,
    )
    stages.append(lr)

    pipeline = Pipeline(stages=stages)
    return pipeline, lr


def add_class_weights(train_df, label_col):
    counts = train_df.groupBy(label_col).count().collect()
    d = {int(r[label_col]): r["count"] for r in counts}
    n_pos = float(d.get(1, 1.0))
    n_neg = float(d.get(0, 1.0))
    w_pos = n_neg / (n_pos + 1e-12)
    w_neg = 1.0
    train_w = train_df.withColumn(
        "class_weight",
        F.when(F.col(label_col) == 1, F.lit(w_pos)).otherwise(F.lit(w_neg)).cast("double"),
    )
    meta = {"n_pos": n_pos, "n_neg": n_neg, "w_pos": w_pos, "w_neg": w_neg, "ratio_pos": n_pos / (n_pos + n_neg)}
    return train_w, meta


def evaluate(pred_df, label_col):
    evaluator_roc = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    evaluator_pr = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderPR")
    roc_auc = evaluator_roc.evaluate(pred_df)
    pr_auc = evaluator_pr.evaluate(pred_df)

    rdd = pred_df.select(F.col("prediction").cast("double"), F.col(label_col).cast("double")).rdd.map(tuple)
    mm = MulticlassMetrics(rdd)

    cm_arr = mm.confusionMatrix().toArray().tolist()
    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "accuracy": mm.accuracy,
        "precision_pos": mm.precision(1.0),
        "recall_pos": mm.recall(1.0),
        "f1_pos": mm.fMeasure(1.0),
        "confusion_matrix": cm_arr,
    }
    return metrics


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def main():
    args = parse_args()
    spark = build_spark()

    df = read_data(spark, args)
    df = ensure_binary_numeric(df, args.label)

    feature_cols = [c.strip() for c in args.features.split(",") if c.strip()]

    # Temporal split
    train_years = [int(y) for y in args.train_years.split(",")]
    test_years = [int(y) for y in args.test_years.split(",")]
    train_df, test_df = temporal_split(df, args.date_col, train_years, test_years)

    # Class weights
    train_w, cw_meta = add_class_weights(train_df, args.label)

    # Pipeline and CV
    pipeline, lr = build_pipeline(feature_cols, standardize=args.standardize, label_col=args.label)
    lr = lr.setWeightCol("class_weight")

    param_grid = (
        ParamGridBuilder()
        .addGrid(lr.regParam, [0.0, 0.01, 0.1, 0.5])
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
        .build()
    )
    evaluator = BinaryClassificationEvaluator(labelCol=args.label, metricName="areaUnderROC")

    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=5,
        parallelism=2,
        seed=args.seed,
    )

    model = cv.fit(train_w)
    pred_test = model.transform(test_df)

    # Metrics
    metrics = evaluate(pred_test, args.label)

    # Export aggregated predictions at ZIP level (mean prob across test dates)
    prob1 = F.udf(lambda v: float(v[1]), T.DoubleType())
    export = (
        pred_test
        .withColumn("prob", prob1(F.col("probability")))
        .groupBy(args.zip_col)
        .agg(F.avg("prob").alias("prob_ocurre_siniestro"))
        .select(args.zip_col, "prob_ocurre_siniestro")
    )

    # Output dirs and saves
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(args.output_dir, f"run_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    model.bestModel.save(os.path.join(out_dir, "lr_pipeline_model"))

    meta = {
        "date_col": args.date_col,
        "zip_col": args.zip_col,
        "label": args.label,
        "features": feature_cols,
        "train_years": train_years,
        "test_years": test_years,
        "standardize": args.standardize,
        "class_weight": cw_meta,
        "input": args.input,
        "input_format": args.input_format,
    }
    save_json(meta, os.path.join(out_dir, "meta.json"))
    save_json(metrics, os.path.join(out_dir, "metrics.json"))

    # Save outputs
    export_parquet = os.path.join(out_dir, "predictions_test_by_zip.parquet")
    export_csv = os.path.join(out_dir, "predictions_test_by_zip.csv")
    export.write.mode("overwrite").parquet(export_parquet)
    export.limit(200000).toPandas().to_csv(export_csv, index=False)

    print("[OK] Saved model and artifacts to:", out_dir)
    print("[OK] Metrics:", json.dumps(metrics, indent=2))

    # === Barrido de umbrales y métricas de clasificación (guardar CSV/JSON) ===
    # Reutilizamos el UDF 'prob1' definido arriba para extraer P(y=1)
    pred_for_thr = pred_test.select(
        F.col(args.label).cast("int").alias("y"),
        prob1(F.col("probability")).alias("p")
    )

    def metrics_for_threshold(th: float):
        b = pred_for_thr.select(
            F.col("y").cast("int").alias("y"),
            (F.col("p") >= F.lit(th)).cast("int").alias("yhat")
        )

        # Convierte cada condición a 0/1 y suma
        agg = b.agg(
            F.sum(F.when((F.col("yhat") == 1) & (F.col("y") == 1), 1).otherwise(0)).alias("TP"),
            F.sum(F.when((F.col("yhat") == 1) & (F.col("y") == 0), 1).otherwise(0)).alias("FP"),
            F.sum(F.when((F.col("yhat") == 0) & (F.col("y") == 1), 1).otherwise(0)).alias("FN"),
            F.sum(F.when((F.col("yhat") == 0) & (F.col("y") == 0), 1).otherwise(0)).alias("TN"),
        ).collect()[0]

        TP, FP, FN, TN = [float(agg[c] or 0.0) for c in ("TP", "FP", "FN", "TN")]
        prec = TP / (TP + FP + 1e-12)
        rec = TP / (TP + FN + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        acc = (TP + TN) / (TP + FP + FN + TN + 1e-12)

        return {
            "threshold": th,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "accuracy": acc,
            "TP": TP, "FP": FP, "FN": FN, "TN": TN
        }

    thresholds = (
            [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50] +
            [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98, 0.99]
    )
    thr_rows = [metrics_for_threshold(th) for th in thresholds]

    # Guardar resultados
    save_json(thr_rows, os.path.join(out_dir, "threshold_sweep.json"))
    try:
        import pandas as pd
        import numpy as np
        pd.DataFrame(thr_rows).to_csv(os.path.join(out_dir, "threshold_sweep.csv"), index=False)
    except Exception as _e:
        # opcional: fallback simple a CSV sin pandas
        with open(os.path.join(out_dir, "threshold_sweep.csv"), "w", encoding="utf-8") as f:
            f.write("threshold,precision,recall,f1,accuracy,TP,FP,FN,TN\n")
            for r in thr_rows:
                f.write(",".join(str(r[k]) for k in ["threshold","precision","recall","f1","accuracy","TP","FP","FN","TN"]) + "\n")

    # Selecciones útiles por pantalla
    best_f1 = max(thr_rows, key=lambda r: r["f1"])
    print("[OK] Threshold sweep saved to:", os.path.join(out_dir, "threshold_sweep.csv"))
    print("[INFO] Best by F1:", json.dumps(best_f1, indent=2))

    # Ejemplo de política: mejor recall sujeto a precisión mínima
    min_precision = 0.10  # <- cambia este objetivo según negocio
    candidates = [r for r in thr_rows if r["precision"] >= min_precision]
    if candidates:
        best_recall_at_min_prec = max(candidates, key=lambda r: r["recall"])
        print(f"[INFO] Best recall with precision >= {min_precision:.2f}:",
              json.dumps(best_recall_at_min_prec, indent=2))
    else:
        print(f"[WARN] No threshold met precision >= {min_precision:.2f}")


    spark.stop()


if __name__ == "__main__":
    main()
