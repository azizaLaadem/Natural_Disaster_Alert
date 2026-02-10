"""
Feature Engineering for Flood Prediction
"""
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from processing.config import *
from processing.utils import (create_spark_session, stop_spark_session, 
                               save_as_single_csv, create_directories)
from pyspark.sql.functions import col, when

def create_interaction_features(df):
    """Create interaction features between key variables"""
    print("\nCreating interaction features...")
    
    df = df.withColumn(
        "rain_water_interaction",
        col("rainfall_mm") * col("water_level_m")
    )
    print("  ✓ rain_water_interaction = rainfall_mm × water_level_m")
    
    df = df.withColumn(
        "discharge_waterlevel",
        col("river_discharge") * col("water_level_m")
    )
    print("  ✓ discharge_waterlevel = river_discharge × water_level_m")
    
    df = df.withColumn(
        "rainfall_discharge",
        col("rainfall_mm") * col("river_discharge")
    )
    print("  ✓ rainfall_discharge = rainfall_mm × river_discharge")
    
    return df

def create_ratio_features(df):
    """Create ratio features"""
    print("\nCreating ratio features...")
    
    df = df.withColumn(
        "discharge_per_rain",
        col("river_discharge") / (col("rainfall_mm") + 1)
    )
    print("  ✓ discharge_per_rain = river_discharge / (rainfall_mm + 1)")
    
    df = df.withColumn(
        "elevation_water_ratio",
        col("elevation_m") / (col("water_level_m") + 0.1)
    )
    print("  ✓ elevation_water_ratio = elevation_m / (water_level_m + 0.1)")
    
    df = df.withColumn(
        "humidity_temp_ratio",
        col("humidity_pct") / (col("temperature_c") + 1)
    )
    print("  ✓ humidity_temp_ratio = humidity_pct / (temperature_c + 1)")
    
    return df

def create_composite_features(df):
    """Create composite risk indices"""
    print("\nCreating composite features...")
    
    df = df.withColumn(
        "flood_risk_index",
        (col("rainfall_mm") / 100) * 0.3 +
        (col("water_level_m") / 10) * 0.3 +
        (col("river_discharge") / 1000) * 0.2 +
        (col("historical_floods") / 10) * 0.2
    )
    print("  ✓ flood_risk_index = weighted combination of key factors")
    
    return df

def create_binary_features(df):
    """Create binary indicator features for extreme conditions"""
    print("\nCreating binary indicator features...")
    
    df = df.withColumn(
        "critical_rainfall",
        when(col("rainfall_mm") > 150, 1).otherwise(0)
    )
    print("  ✓ critical_rainfall = 1 if rainfall > 150mm")
    
    df = df.withColumn(
        "critical_water",
        when(col("water_level_m") > 8, 1).otherwise(0)
    )
    print("  ✓ critical_water = 1 if water_level > 8m")
    
    df = df.withColumn(
        "low_elevation",
        when(col("elevation_m") < 50, 1).otherwise(0)
    )
    print("  ✓ low_elevation = 1 if elevation < 50m")
    
    df = df.withColumn(
        "high_discharge",
        when(col("river_discharge") > 2000, 1).otherwise(0)
    )
    print("  ✓ high_discharge = 1 if discharge > 2000 m³/s")
    
    df = df.withColumn(
        "extreme_event",
        when(
            (col("rainfall_mm") > 200) | 
            (col("water_level_m") > 10) |
            (col("river_discharge") > 3000),
            1
        ).otherwise(0)
    )
    print("  ✓ extreme_event = 1 if any extreme condition met")
    
    return df

def engineer_flood_features():
    """Main function to engineer flood features"""
    print("="*60)
    print("FLOOD FEATURE ENGINEERING")
    print("="*60)
    
    # Create Spark session
    spark = create_spark_session(
        app_name=SPARK_CONFIG["app_name"] + " - Feature Engineering",
        master=SPARK_CONFIG["master"],
        driver_memory=SPARK_CONFIG["driver_memory"],
        executor_memory=SPARK_CONFIG["executor_memory"]
    )
    
    print(f"\n✓ Spark session created (version {spark.version})")
    
    try:
        # Create necessary directories
        create_directories(
            os.path.dirname(FLOOD_FEATURES_PATH),
            os.path.dirname(FLOOD_GOLD_PATH)
        )
        
        # Read cleaned data
        print(f"\n{'='*60}")
        print("Reading cleaned data...")
        print(f"{'='*60}")
        print(f"Source: {FLOOD_CLEANED_PATH}")
        
        df_clean = spark.read.parquet(FLOOD_CLEANED_PATH)
        
        print(f"✓ Data loaded: {df_clean.count()} rows")
        print(f"✓ Original features: {len(df_clean.columns)}")
        
        # Apply feature engineering
        print(f"\n{'='*60}")
        print("Applying feature engineering transformations...")
        print(f"{'='*60}")
        
        df_engineered = df_clean
        
        # Create different types of features
        df_engineered = create_interaction_features(df_engineered)
        df_engineered = create_ratio_features(df_engineered)
        df_engineered = create_composite_features(df_engineered)
        df_engineered = create_binary_features(df_engineered)
        
        print(f"\n✓ Feature engineering complete")
        print(f"✓ Total features: {len(df_engineered.columns)}")
        
        # Cache engineered data
        df_engineered.cache()
        
        # Save full engineered dataset
        print(f"\n{'='*60}")
        print("Saving engineered features...")
        print(f"{'='*60}")
        print(f"Destination: {FLOOD_FEATURES_PATH}")
        
        df_engineered.write \
            .mode("overwrite") \
            .parquet(FLOOD_FEATURES_PATH)
        
        print("✓ Full feature set saved as Parquet")
        
        # Prepare ML dataset (select only necessary columns)
        print(f"\nPreparing ML dataset...")
        ml_columns = FLOOD_ML_FEATURES + [FLOOD_TARGET_COLUMN]
        df_ml = df_engineered.select(*ml_columns)
        
        print(f"✓ ML features selected: {len(FLOOD_ML_FEATURES)}")
        print("\nML Features:")
        for i, feat in enumerate(FLOOD_ML_FEATURES, 1):
            print(f"  {i}. {feat}")
        
        # Save as CSV for ML training
        print(f"\n{'='*60}")
        print("Saving ML dataset...")
        print(f"{'='*60}")
        print(f"Destination: {FLOOD_GOLD_PATH}")
        
        # Save as single CSV
        save_as_single_csv(df_ml, FLOOD_GOLD_PATH)
        
        # Show sample data
        print("\nSample engineered features:")
        df_ml.show(5, truncate=False)
        
        # Show statistics for new features
        print("\nNew feature statistics:")
        new_features = [
            "rain_water_interaction",
            "discharge_per_rain", 
            "flood_risk_index",
            "extreme_event"
        ]
        df_ml.select(new_features).describe().show()
        
        # Show class distribution
        print("\nClass distribution:")
        df_ml.groupBy(FLOOD_TARGET_COLUMN).count().show()
        
        print(f"\n{'='*60}")
        print("FEATURE ENGINEERING COMPLETE")
        print(f"{'='*60}")
        print(f"✓ Engineered features: {df_engineered.count()} rows")
        print(f"✓ Full dataset saved to: {FLOOD_FEATURES_PATH}")
        print(f"✓ ML dataset saved to: {FLOOD_GOLD_PATH}")
        print(f"✓ Ready for model training")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during feature engineering: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        stop_spark_session(spark)
        print("\n✓ Spark session stopped")

if __name__ == "__main__":
    success = engineer_flood_features()
    sys.exit(0 if success else 1)