"""
===============================================================================
Fire Prediction → Kafka → Spark → InfluxDB → Grafana
EXACTLY ALIGNED WITH TRAINING FEATURE ENGINEERING
===============================================================================
"""

import os
import sys
import pytz
from datetime import datetime
from pyspark.sql.functions import col

# =============================================================================
# PATHS & PROJECT
# =============================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

from processing.config import FIRES_MODEL_PATH

# =============================================================================
# SPARK
# =============================================================================
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, when, current_timestamp
from pyspark.sql.types import *
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler

# =============================================================================
# INFLUXDB
# =============================================================================
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# =============================================================================
# CONFIG
# =============================================================================
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "environmental_stream"

MODEL_PATH = os.path.join(FIRES_MODEL_PATH, "random_forest")
FEATURE_NAMES_PATH = os.path.join(FIRES_MODEL_PATH, "feature_names.txt")

INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = "YOUR_TOKEN_HERE"
INFLUXDB_ORG = "fire-monitoring"
INFLUXDB_BUCKET = "fire_predictions"

CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "fires_influx")

# =============================================================================
# SCHEMA (Kafka → Spark)
# =============================================================================
schema = StructType([
    StructField("location_id", StringType()),
    StructField("c_latitude", DoubleType()),
    StructField("c_longitude", DoubleType()),
    StructField("timestamp", StringType()),
    StructField("TEMP_ave", DoubleType()),
    StructField("TEMP_min", DoubleType()),
    StructField("TEMP_max", DoubleType()),
    StructField("PRCP", DoubleType()),
    StructField("SNOW", DoubleType()),
    StructField("WDIR_ave", DoubleType()),
    StructField("WSPD_ave", DoubleType()),
    StructField("PRES_ave", DoubleType()),
    StructField("WCOMP", DoubleType()),
    StructField("ELEV_max", DoubleType()),
    StructField("ELEV_min", DoubleType()),
    StructField("ELEV_median", DoubleType()),
    StructField("ELEV_mean", DoubleType()),
    StructField("SLP_max", DoubleType()),
    StructField("SLP_min", DoubleType()),
    StructField("SLP_median", DoubleType()),
    StructField("SLP_mean", DoubleType()),
    StructField("EVT_mean", DoubleType()),
    StructField("EVH_mean", DoubleType()),
    StructField("EVC_mean", DoubleType()),
    StructField("CBD_mean", DoubleType()),
    StructField("CBH_mean", DoubleType()),
    StructField("CC_mean", DoubleType()),
    StructField("CH_median", DoubleType()),
    StructField("Neighbour_ELEV_mean", DoubleType()),
    StructField("Neighbour_SLP_mean", DoubleType()),
    StructField("Neighbour_EVC_mean", DoubleType()),
])

# =============================================================================
# FEATURE ENGINEERING - EXACTLY AS IN TRAINING
# =============================================================================
def create_features(df):

    print("\n  Creating features...")

    df = df.fillna({"PRCP": 0, "SNOW": 0})

    # ==================== RATIO FEATURES ====================
    print("      - Ratio features...")

    if "ELEV_mean" in df.columns and "Neighbour_ELEV_mean" in df.columns:
        df = df.withColumn("ELEV_ratio", col("ELEV_mean") / (col("Neighbour_ELEV_mean") + 1e-6))

    if "SLP_mean" in df.columns and "Neighbour_SLP_mean" in df.columns:
        df = df.withColumn("SLP_ratio", col("SLP_mean") / (col("Neighbour_SLP_mean") + 1e-6))

    # ==================== RANGE FEATURES ====================
    print("      - Range features...")

    if "ELEV_max" in df.columns and "ELEV_min" in df.columns:
        df = df.withColumn("ELEV_range", col("ELEV_max") - col("ELEV_min"))

    if "SLP_max" in df.columns and "SLP_min" in df.columns:
        df = df.withColumn("SLP_range", col("SLP_max") - col("SLP_min"))

    # ==================== WEATHER INDICES ====================
    print("      - Weather indices...")

    df = df.withColumn(
        "fire_weather_index",
        col("TEMP_max") * col("WSPD_ave") / (col("PRCP") + 1)
    )

    df = df.withColumn(
        "dryness_index",
        col("TEMP_ave") / (col("PRCP") + 1)
    )
    df = df.withColumn(
    "Neighbour_acq_time",
    col("timestamp")   # ou lit(None)
)

    # ==================== INTERACTION FEATURES ====================
    print("      - Interaction features...")

    df = df.withColumn(
        "veg_temp_interaction",
        col("CH_median") * col("TEMP_max")
    )

    df = df.withColumn(
        "lat_lon_interaction",
        col("c_latitude") * col("c_longitude")
    )

    # ==================== BATCH FEATURES (FROM TRAINING) ====================
    print("      - Batch features (diff, ratio, range)...")

    cols = ["ELEV", "SLP", "EVT", "EVH", "EVC", "CBD", "CBH", "CC"]

    for c in cols:
        if f"{c}_mean" in df.columns and f"Neighbour_{c}_mean" in df.columns:
            df = df.withColumn(f"{c}_diff", col(f"{c}_mean") - col(f"Neighbour_{c}_mean"))
            df = df.withColumn(f"{c}_ratio", col(f"{c}_mean") / (col(f"Neighbour_{c}_mean") + 1e-6))

            # ==================== ✅ CORRECTION UNIQUE ====================
            # Feature présente à l'entraînement mais absente en streaming
            df = df.withColumn(f"{c}_sum", col(f"{c}_mean") + col(f"Neighbour_{c}_mean"))
            # ============================================================

        if f"{c}_max" in df.columns and f"{c}_min" in df.columns:
            df = df.withColumn(f"{c}_range", col(f"{c}_max") - col(f"{c}_min"))

    print("      ✅ All features created")
    return df

# =============================================================================
# ALERT LOGIC
# =============================================================================
def create_alerts(df):

    df = df.withColumn(
        "alert_level",
        when(col("prediction") == 2, "HIGH")
        .when(col("prediction") == 1, "MEDIUM")
        .otherwise("LOW")
    )

    df = df.withColumn(
        "alert_level_num",
        when(col("alert_level") == "HIGH", 3)
        .when(col("alert_level") == "MEDIUM", 2)
        .otherwise(1)
    )

    return df

# =============================================================================
# INFLUXDB CLIENT
# =============================================================================
_influx_client = None
_write_api = None

def get_influx_client():
    global _influx_client, _write_api
    if _influx_client is None:
        _influx_client = InfluxDBClient(
            url=INFLUXDB_URL,
            token=INFLUXDB_TOKEN,
            org=INFLUXDB_ORG,
            timeout=30000
        )
        _write_api = _influx_client.write_api(write_options=SYNCHRONOUS)
    return _influx_client, _write_api

# =============================================================================
# WRITE TO INFLUXDB
# =============================================================================
def write_to_influxdb(batch_df, batch_id):

    print(f"\n📊 Processing batch {batch_id}...")

    pdf = batch_df.limit(1000).toPandas()
    if pdf.empty:
        print("   ⚠️ Empty batch")
        return

    client, write_api = get_influx_client()
    points = []

    local_tz = pytz.timezone("Africa/Casablanca")

    for _, row in pdf.iterrows():
        ts = local_tz.localize(row["processing_time"]).astimezone(pytz.UTC)

        point = (
            Point("fire_prediction")
            .tag("location_id", str(row["location_id"]))
            .tag("alert_level", row["alert_level"])
            .field("prediction", int(row["prediction"]))
            .field("alert_level_num", int(row["alert_level_num"]))
            .field("fire_weather_index", float(row["fire_weather_index"]))
            .field("dryness_index", float(row["dryness_index"]))
            .field("latitude", float(row["c_latitude"]))
            .field("longitude", float(row["c_longitude"]))
            .time(ts)
        )
        points.append(point)

    write_api.write(bucket=INFLUXDB_BUCKET, record=points)
    print(f"   💾 Written {len(points)} points")

# =============================================================================
# MAIN
# =============================================================================
def main():

    spark = (
        SparkSession.builder
        .appName("Fire Prediction Streaming")
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    model = RandomForestClassificationModel.load(MODEL_PATH)

    with open(FEATURE_NAMES_PATH) as f:
        FEATURE_COLS = [line.strip() for line in f]

    df = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", KAFKA_TOPIC)
        .load()
    )

    df = (
        df.select(from_json(col("value").cast("string"), schema).alias("data"))
          .select("data.*")
          .withColumn("processing_time", current_timestamp())
    )

    df = create_features(df)
    

    print(" Colonnes disponibles :", df.columns)

    missing = set(FEATURE_COLS) - set(df.columns)
    if missing:
     print(" Features manquantes supprimées :", missing)

    FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]


    assembler = VectorAssembler(
        inputCols=FEATURE_COLS,
        outputCol="features",
        handleInvalid="skip"
    )

    df = assembler.transform(df)
    df = model.transform(df)
    df = create_alerts(df)

    query = (
        df.writeStream
        .foreachBatch(write_to_influxdb)
        .trigger(processingTime="15 seconds")
        .start()
    )

    query.awaitTermination()

# =============================================================================
if __name__ == "__main__":
    main()
