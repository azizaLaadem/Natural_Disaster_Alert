"""
Feature Engineering Script for Fires Dataset
Creates features and target variable for ML models
"""

import sys
import os
from datetime import datetime

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)
sys.path.append(PROJECT_ROOT)

from processing.config import *
from processing.utils import *

from pyspark.sql.functions import col, when
from pyspark.sql.types import DoubleType


def engineer_fires_features():
    """Create features for fires prediction"""
    start_time = datetime.now()

    print("=" * 80)
    print("FIRES FEATURE ENGINEERING")
    print("=" * 80)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Create Spark session
    spark = create_spark_session(
        app_name="Fires Feature Engineering",
        master="local[*]",
        driver_memory="8g",
        executor_memory="8g"
    )

    try:
        # ==================== LOAD DATA ====================
        print("\n" + "=" * 80)
        print("Loading cleaned data")
        print("=" * 80)
        print(f"Source: {FIRES_CLEANED_PATH}")

        df = spark.read.parquet(FIRES_CLEANED_PATH)
        initial_count = df.count()
        initial_cols = len(df.columns)

        print(f"Loaded {initial_count:,} rows and {initial_cols} columns")

        df.cache()

        # ==================== STEP 1: TARGET ====================
        print("\n" + "=" * 80)
        print("STEP 1: Creating target variable (frp_class)")
        print("=" * 80)

        df = df.withColumn("frp", col("frp").cast(DoubleType()))

        df = df.withColumn(
            "frp_class",
            when(col("frp") < 800, 0)
            .when(col("frp") < 1500, 1)
            .otherwise(2)
        )

        print("Target variable created: frp_class")
        print("  - Class 0 (low):    FRP < 800")
        print("  - Class 1 (medium): 800 <= FRP < 1500")
        print("  - Class 2 (high):   FRP >= 1500")

        print("\nTarget distribution:")
        df.groupBy("frp_class").count().orderBy("frp_class").show()

        # ==================== STEP 2: RATIO FEATURES ====================
        print("\n" + "=" * 80)
        print("STEP 2: Ratio features")
        print("=" * 80)

        ratio_features = []

        if "ELEV_mean" in df.columns and "Neighbour_ELEV_mean" in df.columns:
            df = df.withColumn(
                "ELEV_ratio",
                col("ELEV_mean") / (col("Neighbour_ELEV_mean") + 1e-6)
            )
            ratio_features.append("ELEV_ratio")

        if "SLP_mean" in df.columns and "Neighbour_SLP_mean" in df.columns:
            df = df.withColumn(
                "SLP_ratio",
                col("SLP_mean") / (col("Neighbour_SLP_mean") + 1e-6)
            )
            ratio_features.append("SLP_ratio")

        print(f"Created {len(ratio_features)} ratio features")

        # ==================== STEP 3: RANGE FEATURES ====================
        print("\n" + "=" * 80)
        print("STEP 3: Range features")
        print("=" * 80)

        range_features = []

        if "ELEV_max" in df.columns and "ELEV_min" in df.columns:
            df = df.withColumn("ELEV_range", col("ELEV_max") - col("ELEV_min"))
            range_features.append("ELEV_range")

        if "SLP_max" in df.columns and "SLP_min" in df.columns:
            df = df.withColumn("SLP_range", col("SLP_max") - col("SLP_min"))
            range_features.append("SLP_range")

        print(f"Created {len(range_features)} range features")

        # ==================== STEP 4: WEATHER INDICES ====================
        print("\n" + "=" * 80)
        print("STEP 4: Weather indices")
        print("=" * 80)

        weather_features = []

        if all(c in df.columns for c in ["TEMP_max", "WSPD_ave", "PRCP"]):
            df = df.withColumn(
                "fire_weather_index",
                col("TEMP_max") * col("WSPD_ave") / (col("PRCP") + 1)
            )
            weather_features.append("fire_weather_index")

        if "TEMP_ave" in df.columns and "PRCP" in df.columns:
            df = df.withColumn(
                "dryness_index",
                col("TEMP_ave") / (col("PRCP") + 1)
            )
            weather_features.append("dryness_index")

        print(f"Created {len(weather_features)} weather features")

        # ==================== STEP 5: INTERACTIONS ====================
        print("\n" + "=" * 80)
        print("STEP 5: Interaction features")
        print("=" * 80)

        interaction_features = []

        if "CH_median" in df.columns and "TEMP_max" in df.columns:
            df = df.withColumn(
                "veg_temp_interaction",
                col("CH_median") * col("TEMP_max")
            )
            interaction_features.append("veg_temp_interaction")

        if "c_latitude" in df.columns and "c_longitude" in df.columns:
            df = df.withColumn(
                "lat_lon_interaction",
                col("c_latitude") * col("c_longitude")
            )
            interaction_features.append("lat_lon_interaction")

        print(f"Created {len(interaction_features)} interaction features")

        # ==================== STEP 6: BATCH FEATURES ====================
        print("\n" + "=" * 80)
        print("STEP 6: Batch features")
        print("=" * 80)

        batch_features = []
        cols = ["ELEV", "SLP", "EVT", "EVH", "EVC", "CBD", "CBH", "CC"]

        for c in cols:
            if f"{c}_mean" in df.columns and f"Neighbour_{c}_mean" in df.columns:
                df = df.withColumn(
                    f"{c}_diff",
                    col(f"{c}_mean") - col(f"Neighbour_{c}_mean")
                )
                df = df.withColumn(
                    f"{c}_ratio",
                    col(f"{c}_mean") / (col(f"Neighbour_{c}_mean") + 1e-6)
                )
                batch_features.extend([f"{c}_diff", f"{c}_ratio"])

            if f"{c}_max" in df.columns and f"{c}_min" in df.columns:
                df = df.withColumn(
                    f"{c}_range",
                    col(f"{c}_max") - col(f"{c}_min")
                )
                batch_features.append(f"{c}_range")

        print(f"Created {len(batch_features)} batch features")

        # ==================== SAVE ====================
        print("\n" + "=" * 80)
        print("Saving engineered features")
        print("=" * 80)
        print(f"Destination: {FIRES_FEATURES_PATH}")

        df.write.mode("overwrite").parquet(FIRES_FEATURES_PATH)
        # ==================== SAVE FEATURE NAMES ====================
        excluded = {"frp", "frp_class"}

        feature_cols = [c for c in df.columns if c not in excluded]

        feature_names_path = os.path.join(FIRES_MODEL_PATH, "feature_names.txt")

        with open(feature_names_path, "w") as f:
         for c in feature_cols:
          f.write(c + "\n")

        print(f"saved {len(feature_cols)} feature names to {feature_names_path}")


        end_time = datetime.now()

        print_pipeline_summary(
            stage="Feature Engineering",
            success=True,
            start_time=start_time,
            end_time=end_time,
            rows=initial_count,
            initial_columns=initial_cols,
            final_columns=len(df.columns),
            new_features=len(ratio_features + range_features + weather_features +
                             interaction_features + batch_features),
            output_path=FIRES_FEATURES_PATH
        )

        return True

    except Exception as e:
        print("\nERROR during feature engineering:")
        print(str(e))
        import traceback
        traceback.print_exc()
        return False

    finally:
        stop_spark_session(spark)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("\nStarting Fires Feature Engineering...")
    success = engineer_fires_features()
    sys.exit(0 if success else 1)
