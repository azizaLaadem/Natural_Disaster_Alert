import os
import sys

# Configuration Hadoop AVANT tout import PySpark
os.environ['HADOOP_HOME'] = r'C:\hadoop'
os.environ['PATH'] = r'C:\hadoop\bin;' + os.environ.get('PATH', '')

# IMPORTANT: Désactiver les vérifications de permissions
os.environ['SPARK_LOCAL_DIRS'] = os.path.abspath('spark-temp')
os.makedirs('spark-temp', exist_ok=True)

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import *

# Configuration Spark avec options supplémentaires
spark = SparkSession.builder \
    .appName("Flood Streaming Ingestion") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:3.5.1") \
    .config("spark.local.dir", os.path.abspath("spark-temp")) \
    .config("spark.sql.warehouse.dir", os.path.abspath("spark-warehouse")) \
    .config("spark.driver.host", "localhost") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

schema = StructType([
    StructField("location_id", StringType()),
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

print("Connecting to Kafka...")

df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "flood_stream") \
    .option("startingOffsets", "latest") \
    .load()

print("Parsing data...")

parsed = df.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*")

print("Starting stream to console...")

# Créer les répertoires
os.makedirs(r"data\ingested\streaming\flood", exist_ok=True)
os.makedirs(r"data\checkpoints\flood", exist_ok=True)

# Sauvegarder en parquet
query = parsed.writeStream \
    .format("parquet") \
    .option("path", os.path.abspath("data/ingested/streaming/flood")) \
    .option("checkpointLocation", os.path.abspath("data/checkpoints/flood")) \
    .outputMode("append") \
    .start()

print("Stream started! Writing to parquet files...")
query.awaitTermination()                                                                                     