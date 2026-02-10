"""
Batch Ingestion Script for Fires Dataset
Reads CSV and converts to Parquet format
"""
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from processing.config import *
from processing.utils import (
    create_spark_session, 
    stop_spark_session, 
    create_directories,
    check_null_values,
    get_numeric_columns,
    print_statistics,
    calculate_file_size,
    print_pipeline_summary
)

def ingest_fires_data():
    """Ingest fires data from raw CSV to Parquet"""
    start_time = datetime.now()
    
    print("="*80)
    print("FIRES DATA BATCH INGESTION")
    print("="*80)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create Spark session
    spark = create_spark_session(
        app_name="Fires Data Ingestion",
        master="local[*]",
        driver_memory="8g",
        executor_memory="8g"
    )
    
    try:
        # Create necessary directories
        print(f"\n{'='*80}")
        print("Creating directories...")
        print(f"{'='*80}")
        
        create_directories(
            FIRES_INGESTED_PATH,
            os.path.dirname(FIRES_CLEANED_PATH),
            os.path.dirname(FIRES_FEATURES_PATH),
            os.path.dirname(FIRES_GOLD_PATH)
        )
        
        # Check if raw data exists
        print(f"\n{'='*80}")
        print("Checking raw data file...")
        print(f"{'='*80}")
        print(f"Source: {FIRES_RAW_PATH}")
        
        if not os.path.exists(FIRES_RAW_PATH):
            raise FileNotFoundError(f"Raw data file not found: {FIRES_RAW_PATH}")
        
        file_size = calculate_file_size(FIRES_RAW_PATH)
        print(f"File found (Size: {file_size:.2f} MB)")
        
        # Read raw data
        print(f"\n{'='*80}")
        print("Reading raw fires data...")
        print(f"{'='*80}")
        
        df_fires = spark.read \
            .option("header", True) \
            .option("inferSchema", True) \
            .option("multiLine", True) \
            .option("escape", '"') \
            .csv(FIRES_RAW_PATH)
        
        row_count = df_fires.count()
        col_count = len(df_fires.columns)
        
        print(f" Data loaded successfully")
        print(f"  - Rows: {row_count:,}")
        print(f"  - Columns: {col_count}")
        
        # Show schema
        print("\nSchema Information:")
        print("-" * 80)
        df_fires.printSchema()
        
        # Show original columns with types
        print("\nColumn Details:")
        print("-" * 80)
        for i, (col_name, col_type) in enumerate(zip(df_fires.columns, [f.dataType for f in df_fires.schema.fields]), 1):
            print(f"  {i:3d}. {col_name:40s} | Type: {col_type}")
        
        # Cache the dataframe for multiple operations
        df_fires.cache()
        
        # Show sample data
        print(f"\n{'='*80}")
        print("Sample Data (first 5 rows):")
        print(f"{'='*80}")
        df_fires.show(5, truncate=True)
        
        # Data quality checks
        print(f"\n{'='*80}")
        print("Data Quality Analysis:")
        print(f"{'='*80}")
        
        # Check for null values
        null_counts = check_null_values(df_fires)
        
        # Check for target column (if exists)
        possible_targets = ["Fire Occurred", "fire_occurred", "target", "label", "Fire", "frp"]
        target_col = None
        for col_name in possible_targets:
            if col_name in df_fires.columns:
                target_col = col_name
                break
        
        if target_col:
            print(f"\n Target Variable Found: {target_col}")
            df_fires.groupBy(target_col).count() \
                .withColumnRenamed("count", "Count") \
                .orderBy(target_col) \
                .show()
        
        # Numeric statistics
        print(f"\n{'='*80}")
        print("Numeric Statistics:")
        print(f"{'='*80}")
        
        numeric_cols = get_numeric_columns(df_fires)
        
        if numeric_cols:
            print(f"\nFound {len(numeric_cols)} numeric columns")
            print_statistics(df_fires, numeric_cols, max_cols=10)
        
        # Save as Parquet
        print(f"\n{'='*80}")
        print("Saving ingested data...")
        print(f"{'='*80}")
        print(f"Destination: {FIRES_INGESTED_PATH}")
        
        df_fires.write \
            .mode("overwrite") \
            .parquet(FIRES_INGESTED_PATH)
        
        print("Data saved as Parquet format")
        
        # Verify saved data
        print("\nVerifying saved data...")
        df_verify = spark.read.parquet(FIRES_INGESTED_PATH)
        verify_count = df_verify.count()
        
        if verify_count == row_count:
            print(f"Verification successful: {verify_count:,} rows")
        else:
            print(f"Row count mismatch: Original={row_count:,}, Saved={verify_count:,}")
        
        # Calculate Parquet size
        parquet_size = calculate_file_size(FIRES_INGESTED_PATH)
        compression_ratio = (file_size / parquet_size) if parquet_size > 0 else 0
        
        # Summary
        end_time = datetime.now()
        
        print_pipeline_summary(
            stage="Data Ingestion",
            success=True,
            start_time=start_time,
            end_time=end_time,
            original_csv_size_mb=f"{file_size:.2f}",
            parquet_size_mb=f"{parquet_size:.2f}",
            compression_ratio=f"{compression_ratio:.2f}x",
            total_rows=row_count,
            total_columns=col_count,
            numeric_columns=len(numeric_cols),
            null_columns=len(null_counts),
            output_path=FIRES_INGESTED_PATH
        )
        
        return True
        
    except FileNotFoundError as e:
        print(f"\n File Error: {str(e)}")
        print("\n Please ensure the CSV file exists at:")
        print(f"  {FIRES_RAW_PATH}")
        return False
        
    except Exception as e:
        print(f"\n Error during ingestion: {str(e)}")
        print("\nFull traceback:")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Unpersist cached data
        if 'df_fires' in locals():
            df_fires.unpersist()
        
        stop_spark_session(spark)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    print("\nStarting Fires Data Batch Ingestion Pipeline...")
    success = ingest_fires_data()
    
    if success:
        print("\nIngestion completed successfully!")
        sys.exit(0)
    else:
        print("\n Ingestion failed!")
        sys.exit(1)