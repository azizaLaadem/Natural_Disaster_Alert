
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from processing.config import *
from processing.utils import create_spark_session, stop_spark_session, create_directories

def ingest_flood_data():
    """Ingest flood data from raw CSV to Parquet"""
    print("="*60)
    print("FLOOD DATA BATCH INGESTION")
    print("="*60)
    
    # Create Spark session
    spark = create_spark_session(
        app_name=SPARK_CONFIG["app_name"] + " - Ingestion",
        master=SPARK_CONFIG["master"],
        driver_memory=SPARK_CONFIG["driver_memory"],
        executor_memory=SPARK_CONFIG["executor_memory"]
    )
    
    print(f"\n✓ Spark session created (version {spark.version})")
    
    try:
        # Create necessary directories
        create_directories(
            os.path.dirname(FLOOD_INGESTED_PATH),
            os.path.dirname(FLOOD_CLEANED_PATH),
            os.path.dirname(FLOOD_FEATURES_PATH),
            os.path.dirname(FLOOD_GOLD_PATH)
        )
        
        # Read raw data
        print(f"\n{'='*60}")
        print("Reading raw flood data...")
        print(f"{'='*60}")
        print(f"Source: {FLOOD_RAW_PATH}")
        
        if not os.path.exists(FLOOD_RAW_PATH):
            raise FileNotFoundError(f"Raw data file not found: {FLOOD_RAW_PATH}")
        
        df_flood = spark.read \
            .option("header", True) \
            .option("inferSchema", True) \
            .csv(FLOOD_RAW_PATH)
        
        row_count = df_flood.count()
        col_count = len(df_flood.columns)
        
        print(f"✓ Data loaded successfully")
        print(f"  - Rows: {row_count}")
        print(f"  - Columns: {col_count}")
        
        # Show original columns
        print("\nOriginal columns:")
        for i, col_name in enumerate(df_flood.columns, 1):
            print(f"  {i}. {col_name}")
        
        # Show sample data
        print("\nSample data (first 5 rows):")
        df_flood.show(5, truncate=True)
        
        # Check for target column
        target_col = "Flood Occurred"
        if target_col in df_flood.columns:
            print("\nClass distribution:")
            df_flood.groupBy(target_col).count().show()
        
        # Cache the dataframe
        df_flood.cache()
        
        # Save as Parquet
        print(f"\n{'='*60}")
        print("Saving ingested data...")
        print(f"{'='*60}")
        print(f"Destination: {FLOOD_INGESTED_PATH}")
        
        df_flood.write \
            .mode("overwrite") \
            .parquet(FLOOD_INGESTED_PATH)
        
        print("✓ Data saved as Parquet format")
        
        # Show data statistics
        print("\nData Statistics:")
        numeric_cols = ["Rainfall (mm)", "Temperature (°C)", "Water Level (m)"]
        available_cols = [col for col in numeric_cols if col in df_flood.columns]
        if available_cols:
            df_flood.select(available_cols).describe().show()
        
        print(f"\n{'='*60}")
        print("INGESTION COMPLETE")
        print(f"{'='*60}")
        print(f"✓ Ingested {row_count} rows with {col_count} columns")
        print(f"✓ Saved to: {FLOOD_INGESTED_PATH}")
        print(f"✓ Ready for cleaning step")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during ingestion: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        stop_spark_session(spark)
        print("\n✓ Spark session stopped")

if __name__ == "__main__":
    success = ingest_flood_data()
    sys.exit(0 if success else 1)