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
    builder = (
        SparkSession.builder.appName("tfm-train-logreg")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "200")
    )
    return builder.getOrCreate()


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

    spark.stop()


if __name__ == "__main__":
    main()
