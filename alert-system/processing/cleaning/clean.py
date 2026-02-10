"""
Data Cleaning for Flood Prediction
"""
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from processing.config import *
from processing.utils import (create_spark_session, stop_spark_session, 
                               check_null_values, create_directories)
from pyspark.sql.functions import col

def clean_column_names(df):
    """Clean and normalize column names"""
    print("\nCleaning column names...")
    df_clean = df.toDF(*[c.lower().strip() for c in df.columns])
    print("Column names normalized (lowercase, trimmed)")
    return df_clean

def rename_columns(df, column_mapping):
    """Rename columns based on mapping"""
    print("\nRenaming columns to standard format...")
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df = df.withColumnRenamed(old_name, new_name)
            print(f"   {old_name} → {new_name}")
    return df

def handle_missing_values(df, numeric_cols, categorical_cols):
    """Handle missing values in numeric and categorical columns"""
    print("\nHandling missing values...")
    
    # Check for nulls
    null_counts = check_null_values(df)
    
    # Fill numeric columns with median
    print("\nFilling numeric columns with median:")
    for col_name in numeric_cols:
        if col_name in df.columns:
            null_count = null_counts.get(col_name, 0)
            if null_count > 0:
                median = df.approxQuantile(col_name, [0.5], 0.01)[0]
                df = df.fillna({col_name: median})
                print(f"   {col_name}: filled {null_count} nulls with median {median:.2f}")
            else:
                print(f"   {col_name}: no nulls")
    
    # Fill categorical columns with mode
    print("\nFilling categorical columns with mode:")
    for col_name in categorical_cols:
        if col_name in df.columns:
            null_count = null_counts.get(col_name, 0)
            if null_count > 0:
                mode_row = df.groupBy(col_name).count().orderBy(col("count").desc()).first()
                if mode_row:
                    mode = mode_row[0]
                    df = df.fillna({col_name: mode})
                    print(f"   {col_name}: filled {null_count} nulls with mode '{mode}'")
            else:
                print(f"   {col_name}: no nulls")
    
    return df

def clean_flood_data():
    """Main function to clean flood data"""
    print("="*60)
    print("FLOOD DATA CLEANING")
    print("="*60)
    
    # Create Spark session
    spark = create_spark_session(
        app_name=SPARK_CONFIG["app_name"] + " - Cleaning",
        master=SPARK_CONFIG["master"],
        driver_memory=SPARK_CONFIG["driver_memory"],
        executor_memory=SPARK_CONFIG["executor_memory"]
    )
    
    print(f"\nSpark session created (version {spark.version})")
    
    try:
        # Create necessary directories
        create_directories(
            os.path.dirname(FLOOD_CLEANED_PATH),
            os.path.dirname(FLOOD_GOLD_PATH)
        )
        
        # Read ingested data
        print(f"\n{'='*60}")
        print("Reading ingested data...")
        print(f"{'='*60}")
        print(f"Source: {FLOOD_INGESTED_PATH}")
        
        df_raw = spark.read.parquet(FLOOD_INGESTED_PATH)
        
        print(f" Data loaded: {df_raw.count()} rows")
        
        # Step 1: Clean column names
        df_clean = clean_column_names(df_raw)
        
        # Step 2: Rename columns
        df_renamed = rename_columns(df_clean, FLOOD_COLUMN_MAPPING)
        
        print("\nFinal column names:")
        for col_name in df_renamed.columns:
            print(f"  - {col_name}")
        
        # Step 3: Handle missing values
        df_cleaned = handle_missing_values(
            df_renamed, 
            FLOOD_NUMERIC_FEATURES, 
            FLOOD_CATEGORICAL_FEATURES
        )
        
        # Verify no nulls remain
        print("\nVerifying data quality...")
        final_null_counts = check_null_values(df_cleaned)
        total_nulls = sum(final_null_counts.values())
        
        if total_nulls == 0:
            print("\n All missing values handled successfully")
        else:
            print(f"\n Warning: {total_nulls} null values remain")
        
        # Cache cleaned data
        df_cleaned.cache()
        
        # Save cleaned data
        print(f"\n{'='*60}")
        print("Saving cleaned data...")
        print(f"{'='*60}")
        print(f"Destination: {FLOOD_CLEANED_PATH}")
        
        df_cleaned.write \
            .mode("overwrite") \
            .parquet(FLOOD_CLEANED_PATH)
        
        print("Cleaned data saved as Parquet")
        
        # Show sample
        print("\nSample cleaned data:")
        df_cleaned.show(5, truncate=True)
        
        # Show statistics
        print("\nData statistics:")
        numeric_cols_present = [c for c in FLOOD_NUMERIC_FEATURES if c in df_cleaned.columns]
        if numeric_cols_present:
            df_cleaned.select(numeric_cols_present[:5]).describe().show()
        
        print(f"\n{'='*60}")
        print("CLEANING COMPLETE")
        print(f"{'='*60}")
        print(f" Cleaned data: {df_cleaned.count()} rows")
        print(f"Saved to: {FLOOD_CLEANED_PATH}")
        print(f" Ready for feature engineering")
        
        return True
        
    except Exception as e:
        print(f"\n Error during cleaning: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        stop_spark_session(spark)
        print("\n Spark session stopped")

if __name__ == "__main__":
    success = clean_flood_data()
    sys.exit(0 if success else 1)