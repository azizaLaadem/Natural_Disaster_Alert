"""
Real-Time Flood Prediction Dashboard
Visualize predictions and alerts in real-time
"""
import os
import sys
import time
from datetime import datetime
import json

# Configuration Hadoop
os.environ['HADOOP_HOME'] = r'C:\hadoop'
os.environ['PATH'] = r'C:\hadoop\bin;' + os.environ.get('PATH', '')
os.environ['SPARK_LOCAL_DIRS'] = os.path.abspath('spark-temp')

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, max as spark_max, window

PREDICTIONS_PATH = "data/predictions/streaming/flood"
ALERTS_PATH = "data/alerts/flood"

def clear_screen():
    """Clear console screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print dashboard header"""
    print("="*100)
    print("🌊 REAL-TIME FLOOD PREDICTION DASHBOARD 🌊".center(100))
    print("="*100)
    print(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(100))
    print("="*100)

def print_metrics(spark, predictions_path, alerts_path):
    """Print real-time metrics"""
    
    try:
        # Lire les prédictions
        df_predictions = spark.read.parquet(predictions_path)
        total_predictions = df_predictions.count()
        
        if total_predictions == 0:
            print("\n⏳ Waiting for predictions...\n")
            return
        
        # Statistiques générales
        print("\n📊 OVERALL STATISTICS")
        print("-" * 100)
        print(f"Total Predictions: {total_predictions}")
        
        flood_count = df_predictions.filter(col("prediction") == 1).count()
        no_flood_count = total_predictions - flood_count
        flood_pct = (flood_count / total_predictions * 100) if total_predictions > 0 else 0
        
        print(f"Flood Predicted:   {flood_count} ({flood_pct:.1f}%)")
        print(f"No Flood:          {no_flood_count} ({100-flood_pct:.1f}%)")
        
        # Alert levels
        print("\n🚨 ALERT LEVELS")
        print("-" * 100)
        alert_summary = df_predictions.groupBy("alert_level").count().collect()
        for row in sorted(alert_summary, key=lambda x: x['count'], reverse=True):
            level = row['alert_level']
            count_val = row['count']
            pct = (count_val / total_predictions * 100)
            
            if level == "CRITICAL":
                symbol = "🔴"
            elif level == "HIGH":
                symbol = "🟠"
            elif level == "MODERATE":
                symbol = "🟡"
            else:
                symbol = "🟢"
            
            bar = "█" * int(pct / 2)
            print(f"{symbol} {level:<12} {count_val:>6} ({pct:>5.1f}%) {bar}")
        
        # Top risky locations
        print("\n📍 TOP 10 HIGH-RISK LOCATIONS")
        print("-" * 100)
        high_risk = df_predictions.filter(col("prediction") == 1) \
            .orderBy(col("flood_risk_index").desc()) \
            .select("location_id", "alert_level", "flood_risk_index", 
                   "rainfall_mm", "water_level_m") \
            .limit(10)
        
        high_risk.show(truncate=False)
        
        # Environmental conditions
        print("\n🌡️ CURRENT ENVIRONMENTAL CONDITIONS")
        print("-" * 100)
        
        stats = df_predictions.agg(
            avg("rainfall_mm").alias("avg_rainfall"),
            spark_max("rainfall_mm").alias("max_rainfall"),
            avg("water_level_m").alias("avg_water_level"),
            spark_max("water_level_m").alias("max_water_level"),
            avg("temperature_c").alias("avg_temp")
        ).collect()[0]
        
        print(f"Rainfall:     Avg={stats['avg_rainfall']:.1f}mm, Max={stats['max_rainfall']:.1f}mm")
        print(f"Water Level:  Avg={stats['avg_water_level']:.2f}m, Max={stats['max_water_level']:.2f}m")
        print(f"Temperature:  Avg={stats['avg_temp']:.1f}°C")
        
        # Critical alerts
        if os.path.exists(alerts_path):
            df_alerts = spark.read.parquet(alerts_path)
            critical_count = df_alerts.count()
            
            if critical_count > 0:
                print(f"\n🚨 CRITICAL ALERTS: {critical_count}")
                print("-" * 100)
                recent_alerts = df_alerts.orderBy(col("processing_time").desc()) \
                    .select("location_id", "alert_level", "alert_message", 
                           "rainfall_mm", "water_level_m") \
                    .limit(5)
                recent_alerts.show(truncate=False)
        
        # Recent predictions
        print("\n📋 LATEST PREDICTIONS")
        print("-" * 100)
        recent = df_predictions.orderBy(col("processing_time").desc()) \
            .select("location_id", "prediction", "alert_level", 
                   "flood_risk_index", "processing_time") \
            .limit(10)
        recent.show(truncate=False)
        
    except Exception as e:
        print(f"\n⚠️ Error reading data: {e}")
        print("Make sure the prediction system is running!")

def main():
    print("Starting Real-Time Dashboard...")
    
    # Créer Spark session
    spark = SparkSession.builder \
        .appName("Flood Dashboard") \
        .config("spark.local.dir", os.path.abspath("spark-temp")) \
        .config("spark.sql.warehouse.dir", os.path.abspath("spark-warehouse")) \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    print("\n✓ Dashboard initialized")
    print("✓ Press Ctrl+C to exit\n")
    time.sleep(2)
    
    refresh_interval = 5  # seconds
    
    try:
        while True:
            clear_screen()
            print_header()
            print_metrics(spark, PREDICTIONS_PATH, ALERTS_PATH)
            print("\n" + "="*100)
            print(f"Auto-refresh in {refresh_interval} seconds... (Press Ctrl+C to exit)".center(100))
            print("="*100)
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\n✓ Dashboard stopped")
        spark.stop()

if __name__ == "__main__":
    main()