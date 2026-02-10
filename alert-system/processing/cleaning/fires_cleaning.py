"""
Data Cleaning Script for Fires Dataset
Cleans the ingested data and prepares it for feature engineering
"""
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
)

from processing.config import *
from processing.utils import *
from pyspark.sql.functions import col, mean
from pyspark.sql.types import DoubleType


def clean_fires_data():
    """Clean fires data after ingestion"""
    start_time = datetime.now()

    print("=" * 80)
    print("FIRES DATA CLEANING")
    print("=" * 80)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Create Spark session
    spark = create_spark_session(
        app_name="Fires Data Cleaning",
        master="local[*]",
        driver_memory="8g",
        executor_memory="8g"
    )

    try:
        # Load ingested data
        print("\n" + "=" * 80)
        print("Loading ingested data...")
        print("=" * 80)
        print(f"Source: {FIRES_INGESTED_PATH}")

        df = spark.read.parquet(FIRES_INGESTED_PATH)
        initial_count = df.count()
        initial_cols = len(df.columns)

        print(f"Data loaded: {initial_count} rows, {initial_cols} columns")

        df.cache()

        # ==================== STEP 1: Drop unnecessary columns ====================
        print("\n" + "=" * 80)
        print("STEP 1: Dropping unnecessary columns")
        print("=" * 80)

        cols_to_drop = [
            "Neighbour_c_longitude",
            "acq_date",
            "acq_time",
            "Polygon_ID",
            "Neighbour",
            "CH_mean",
            "Neighbour_CH_mean"
        ]

        cols_to_drop = [c for c in cols_to_drop if c in df.columns]

        if cols_to_drop:
            df = df.drop(*cols_to_drop)
            print(f"Dropped {len(cols_to_drop)} columns:")
            for c in cols_to_drop:
                print(f" - {c}")
        else:
            print("No columns to drop")

        # ==================== STEP 2: Handle missing values ====================
        print("\n" + "=" * 80)
        print("STEP 2: Handling missing values")
        print("=" * 80)

        null_before = check_null_values(df)

        print("\nFilling PRCP and SNOW with 0")
        if "PRCP" in df.columns:
            df = df.fillna({"PRCP": 0})
        if "SNOW" in df.columns:
            df = df.fillna({"SNOW": 0})

        print("\nImputing weather columns with mean")
        impute_mean_cols = [
            "WDIR_ave", "WCOMP", "TEMP_ave",
            "TEMP_min", "TEMP_max", "WSPD_ave", "PRES_ave"
        ]
        impute_mean_cols = [c for c in impute_mean_cols if c in df.columns]

        if impute_mean_cols:
            mean_values = (
                df.select([mean(col(c)).alias(c) for c in impute_mean_cols])
                .collect()[0]
                .asDict()
            )

            df = df.fillna(mean_values)

            for c, v in mean_values.items():
                if v is not None:
                    print(f" - {c}: mean = {v:.2f}")

        # ==================== STEP 3: Process FRP ====================
        print("\n" + "=" * 80)
        print("STEP 3: Processing FRP")
        print("=" * 80)

        if "frp" in df.columns:
            before_filter = df.count()

            df = df.filter(col("frp").rlike("^[0-9]+(\\.[0-9]+)?$"))
            after_filter = df.count()
            removed = before_filter - after_filter

            print(f"FRP filtering:")
            print(f" - Before: {before_filter}")
            print(f" - After: {after_filter}")
            print(f" - Removed: {removed}")

            df = df.withColumn("frp", col("frp").cast(DoubleType()))
            print("FRP converted to Double")
        else:
            print("FRP column not found")

        # ==================== STEP 4: Data quality summary ====================
        print("\n" + "=" * 80)
        print("STEP 4: Data Quality Summary")
        print("=" * 80)

        final_count = df.count()
        final_cols = len(df.columns)

        null_after = check_null_values(df)

        rows_removed = initial_count - final_count
        cols_removed = initial_cols - final_cols

        print("\nBefore cleaning:")
        print(f" - Rows: {initial_count}")
        print(f" - Columns: {initial_cols}")
        print(f" - Null columns: {len(null_before)}")

        print("\nAfter cleaning:")
        print(f" - Rows: {final_count}")
        print(f" - Columns: {final_cols}")
        print(f" - Null columns: {len(null_after)}")

        print("\nChanges:")
        print(f" - Rows removed: {rows_removed}")
        print(f" - Columns removed: {cols_removed}")

        # ==================== STEP 5: Save cleaned data ====================
        print("\n" + "=" * 80)
        print("STEP 5: Saving cleaned data")
        print("=" * 80)
        print(f"Destination: {FIRES_CLEANED_PATH}")

        df.write.mode("overwrite").parquet(FIRES_CLEANED_PATH)

        df_verify = spark.read.parquet(FIRES_CLEANED_PATH)
        verify_count = df_verify.count()

        print(f"Saved rows: {verify_count}")

        end_time = datetime.now()

        print_pipeline_summary(
            stage="Data Cleaning",
            success=True,
            start_time=start_time,
            end_time=end_time,
            initial_rows=initial_count,
            final_rows=final_count,
            rows_removed=rows_removed,
            initial_columns=initial_cols,
            final_columns=final_cols,
            columns_removed=cols_removed,
            output_path=FIRES_CLEANED_PATH
        )

        return True

    except Exception as e:
        print("\nERROR during cleaning")
        print(str(e))
        import traceback
        traceback.print_exc()
        return False

    finally:
        if "df" in locals():
            df.unpersist()
        stop_spark_session(spark)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("\nStarting Fires Data Cleaning Pipeline...")
    success = clean_fires_data()

    if success:
        print("\nCleaning completed successfully")
        sys.exit(0)
    else:
        print("\nCleaning failed")
        sys.exit(1)
