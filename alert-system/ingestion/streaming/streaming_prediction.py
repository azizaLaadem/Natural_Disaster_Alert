"""
===============================================================================
Flood Prediction → InfluxDB → Grafana (TYPE CONFLICT FIXED)
===============================================================================
"""

import os
import sys
import shutil
import pytz
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, when, current_timestamp
from pyspark.sql.types import *
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array

try:
    from influxdb_client import InfluxDBClient, Point
    from influxdb_client.client.write_api import SYNCHRONOUS
except ImportError:
    print("Installing influxdb-client...")
    os.system("pip install influxdb-client")
    from influxdb_client import InfluxDBClient, Point
    from influxdb_client.client.write_api import SYNCHRONOUS

# =============================================================================
# CONFIG
# =============================================================================
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"   
KAFKA_TOPIC = "flood_stream"

MODEL_PATH = r"C:\Users\laade\Natural_Disaster_Alert\alert-system\ml\models\flood_model"

INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = "BdeLJauoFfO2x4vP4jwtT9xBv5KDF4Nh2b5YvMPd7sa8izjhGhHzkF2JF3SfijlvAx7YsWxkAT-DFErvescglQ=="
INFLUXDB_ORG = "flood-monitoring"
INFLUXDB_BUCKET = "flood_predictions"

CHECKPOINT_PATH = r"C:\Users\laade\Natural_Disaster_Alert\alert-system\checkpoints\flood_influx"

# =============================================================================
# GPS COORDINATES
# =============================================================================
LOCATION_COORDS = {
    "LOC_001": {"lat": 33.5731, "lon": -7.5898, "name": "Casablanca"},
    "LOC_002": {"lat": 34.0209, "lon": -6.8416, "name": "Rabat"},
    "LOC_003": {"lat": 31.6295, "lon": -7.9811, "name": "Marrakech"},
    "LOC_004": {"lat": 35.7595, "lon": -5.8340, "name": "Tanger"},
    "LOC_005": {"lat": 33.8893, "lon": -5.5535, "name": "Fès"},
    "LOC_006": {"lat": 30.4278, "lon": -9.5981, "name": "Agadir"},
    "LOC_007": {"lat": 34.6814, "lon": -1.9086, "name": "Oujda"},
    "LOC_008": {"lat": 33.2316, "lon": -8.5007, "name": "El Jadida"},
    "LOC_009": {"lat": 35.1681, "lon": -5.2684, "name": "Tétouan"},
    "LOC_010": {"lat": 32.2949, "lon": -9.2372, "name": "Safi"},
    "LOC_011": {"lat": 34.2572, "lon": -6.5965, "name": "Kénitra"},
    "LOC_012": {"lat": 33.5883, "lon": -7.6114, "name": "Mohammedia"},
    "LOC_013": {"lat": 35.2595, "lon": -3.9366, "name": "Nador"},
    "LOC_014": {"lat": 33.9715, "lon": -6.8498, "name": "Salé"},
    "LOC_015": {"lat": 31.5085, "lon": -9.7595, "name": "Essaouira"},
}

# =============================================================================
# SCHEMA
# =============================================================================
schema = StructType([
    StructField("location_id", StringType()),
    StructField("location_name", StringType()),
    StructField("latitude", DoubleType()),
    StructField("longitude", DoubleType()),
    StructField("temperature_c", DoubleType()),
    StructField("rainfall_mm", DoubleType()),
    StructField("humidity_pct", DoubleType()),
    StructField("river_discharge_m3s", DoubleType()),
    StructField("water_level_m", DoubleType()),
    StructField("elevation_m", IntegerType()),
    StructField("population_density", IntegerType()),
    StructField("historical_floods", IntegerType()),
    StructField("infrastructure", StringType()),
    StructField("soil_type", StringType()),
    StructField("land_cover", StringType()),
    StructField("timestamp", StringType())
])

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def create_features(df):
    """Create ALL features - OPTIMIZED"""
    df = df.withColumnRenamed("river_discharge_m3s", "river_discharge")
    
    df = df.withColumn("rain_water_interaction",
                       col("rainfall_mm") * col("water_level_m"))
    df = df.withColumn("discharge_waterlevel",
                       col("river_discharge") * col("water_level_m"))
    df = df.withColumn("rainfall_discharge",
                       col("rainfall_mm") * col("river_discharge"))
    df = df.withColumn("discharge_per_rain",
                       col("river_discharge") / (col("rainfall_mm") + 1))
    df = df.withColumn("elevation_water_ratio",
                       col("elevation_m") / (col("water_level_m") + 0.1))
    df = df.withColumn("humidity_temp_ratio",
                       col("humidity_pct") / (col("temperature_c") + 1))
    df = df.withColumn(
        "flood_risk_index",
        (col("rainfall_mm") / 100) * 0.3 +
        (col("water_level_m") / 10) * 0.3 +
        (col("river_discharge") / 1000) * 0.2 +
        (col("historical_floods") / 10) * 0.2
    )
    df = df.withColumn("critical_rainfall",
                       when(col("rainfall_mm") > 150, 1).otherwise(0))
    df = df.withColumn("critical_water",
                       when(col("water_level_m") > 8, 1).otherwise(0))
    df = df.withColumn("low_elevation",
                       when(col("elevation_m") < 50, 1).otherwise(0))
    df = df.withColumn("high_discharge",
                       when(col("river_discharge") > 2000, 1).otherwise(0))
    df = df.withColumn(
        "extreme_event",
        when(
            (col("rainfall_mm") > 200) |
            (col("water_level_m") > 10) |
            (col("river_discharge") > 3000),
            1
        ).otherwise(0)
    )
    return df

def create_alerts(df):
    """Create alerts"""
    df = df.withColumn("prob_flood",
                       vector_to_array(col("probability"))[1])
    df = df.withColumn(
        "alert_level",
        when(col("prediction") == 1,
             when(col("prob_flood") > 0.9, "CRITICAL")
             .when(col("prob_flood") > 0.7, "HIGH")
             .otherwise("MODERATE"))
        .otherwise("LOW")
    )
    df = df.withColumn(
        "alert_level_num",
        when(col("alert_level") == "CRITICAL", 4)
        .when(col("alert_level") == "HIGH", 3)
        .when(col("alert_level") == "MODERATE", 2)
        .otherwise(1)
    )
    return df

# =============================================================================
# INFLUXDB CLIENT
# =============================================================================
_influx_client = None
_write_api = None

def get_influx_client():
    """Get InfluxDB client"""
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
# 🔥 FIX: SAFE TYPE CONVERSION HELPERS
# =============================================================================
def safe_int(value):
    """Convert to int safely"""
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return 0

def safe_float(value):
    """Convert to float safely"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

# =============================================================================
# WRITE TO INFLUXDB (TYPE CONFLICT FIXED)
# =============================================================================
def write_to_influxdb(batch_df, batch_id):
    """
    ✅ TYPE CONFLICT FIXED: Explicit type casting for all fields
    """
    
    try:
        print(f"\n{'='*80}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Batch #{batch_id}")
        print('='*80)
        
        pdf = batch_df.limit(100).toPandas()
        
        if len(pdf) == 0:
            print(f"⚠️ Batch {batch_id}: Empty")
            return
        
        print(f"📊 Processing {len(pdf)} records")
        
        client, write_api = get_influx_client()
        
        points = []
        local_tz = pytz.timezone('Africa/Casablanca')
        utc_tz = pytz.UTC
        
        for _, row in pdf.iterrows():
            location_id = row["location_id"]
            coords = LOCATION_COORDS.get(location_id, {"lat": 0.0, "lon": 0.0, "name": "Unknown"})
            
            # Timestamp handling
            processing_time = row["processing_time"]
            if isinstance(processing_time, str):
                timestamp = datetime.fromisoformat(processing_time)
            else:
                timestamp = processing_time.to_pydatetime()
            
            if timestamp.tzinfo is None:
                timestamp = local_tz.localize(timestamp).astimezone(utc_tz)
            else:
                timestamp = timestamp.astimezone(utc_tz)
            
            # 🔥 FIX: Explicit type casting with safe conversions
            point = (
                Point("flood_prediction")
                .tag("location_id", str(location_id))
                .tag("alert_level", str(row["alert_level"]))
                .tag("location_name", str(coords["name"]))
                # Integer fields - MUST be int, not float
                .field("prediction", safe_int(row["prediction"]))
                .field("alert_level_num", safe_int(row["alert_level_num"]))
                .field("elevation_m", safe_int(row["elevation_m"]))
                .field("extreme_event", safe_int(row["extreme_event"]))
                .field("critical_rainfall", safe_int(row.get("critical_rainfall", 0)))
                .field("critical_water", safe_int(row.get("critical_water", 0)))
                .field("low_elevation", safe_int(row.get("low_elevation", 0)))
                .field("high_discharge", safe_int(row.get("high_discharge", 0)))
                # Float fields
                .field("probability", safe_float(row["prob_flood"]))
                .field("flood_risk_index", safe_float(row["flood_risk_index"]))
                .field("latitude", safe_float(coords["lat"]))
                .field("longitude", safe_float(coords["lon"]))
                .field("rainfall_mm", safe_float(row["rainfall_mm"]))
                .field("water_level_m", safe_float(row["water_level_m"]))
                .field("river_discharge", safe_float(row["river_discharge"]))
                .field("temperature_c", safe_float(row["temperature_c"]))
                .field("humidity_pct", safe_float(row["humidity_pct"]))
                .time(timestamp)
            )
            points.append(point)
        
        if points:
            print(f"🔄 Writing {len(points)} points to InfluxDB...")
            write_api.write(bucket=INFLUXDB_BUCKET, record=points)
            print(f"✅ {len(points)} records → InfluxDB")
            
            alerts = pdf[pdf["alert_level"].isin(["CRITICAL", "HIGH"])]
            if len(alerts) > 0:
                print(f"\n⚠️  {len(alerts)} CRITICAL/HIGH alerts detected!")
                for _, alert in alerts.iterrows():
                    print(f"   🔴 {alert['location_name']}: {alert['alert_level']} "
                          f"(prob: {alert['prob_flood']:.1%})")
        
    except Exception as e:
        print(f"❌ Error in batch {batch_id}: {e}")
        import traceback
        traceback.print_exc()

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 80)
    print(" 🌊 FLOOD PREDICTION SYSTEM (TYPE CONFLICT FIXED) ".center(80))
    print("=" * 80)
    
    # Clean checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        response = input(f"\n⚠️  Delete checkpoint directory? (y/n): ")
        if response.lower() == 'y':
            try:
                shutil.rmtree(CHECKPOINT_PATH)
                print("✓ Checkpoint deleted")
            except Exception as e:
                print(f"⚠️  Could not delete checkpoint: {e}")
    
    print("\n🔧 Creating Spark session...")
    spark = (
        SparkSession.builder
        .appName("Flood Prediction System")
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:3.5.1")
        .config("spark.driver.memory", "2g")
        .config("spark.executor.memory", "2g")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.python.worker.reuse", "true")
        .config("spark.python.worker.timeout", "600")
        .config("spark.executor.heartbeatInterval", "60s")
        .config("spark.network.timeout", "600s")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.sql.warehouse.dir", os.path.abspath("spark-warehouse"))
        .config("spark.driver.host", "localhost")
        .getOrCreate()
    )
    
    spark.sparkContext.setLogLevel("ERROR")
    print("✓ Spark session created")
    
    print(f"\n🤖 Loading ML model from: {MODEL_PATH}")
    try:
        model = PipelineModel.load(MODEL_PATH)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    print(f"\n📡 Connecting to Kafka...")
    print(f"   • Bootstrap servers: {KAFKA_BOOTSTRAP_SERVERS}")
    print(f"   • Topic: {KAFKA_TOPIC}")
    
    df = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .option("maxOffsetsPerTrigger", "10")
        .load()
    )
    print("✓ Kafka connected")
    
    print("\n⚙️  Building processing pipeline...")
    df = (
        df.select(from_json(col("value").cast("string"), schema).alias("data"))
          .select("data.*")
          .withColumn("processing_time", current_timestamp())
    )
    
    df = create_features(df)
    df = model.transform(df)
    df = create_alerts(df)
    print("✓ Pipeline ready")
    
    print(f"\n Starting streaming query...")
    query = (
        df.writeStream
        .foreachBatch(write_to_influxdb)
        .option("checkpointLocation", CHECKPOINT_PATH)
        .trigger(processingTime="15 seconds")
        .start()
    )
    
    print("\n" + "=" * 80)
    print(" SYSTEM STARTED - TYPE CONFLICT FIXED!".center(80))
    print("=" * 80)
    print("\n Access your dashboards:")
    print("   InfluxDB UI:  http://localhost:8086")
    print("   Grafana:      http://localhost:3000")
    print("\n Press Ctrl+C to stop gracefully\n")
    
    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        print("\n\nStopping system...")
        query.stop()
        if _influx_client:
            _influx_client.close()
        spark.stop()
        print("System stopped gracefully")

if __name__ == "__main__":
    main()