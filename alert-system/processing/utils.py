"""
Spark utility functions for Natural Disaster Alert System
"""
import os
import sys
from datetime import datetime
from pyspark.sql import SparkSession


def create_spark_session(app_name="Spark App", master="local[*]",
                         driver_memory="4g", executor_memory="4g"):
    """
    Create and configure a Spark session
    """
    # Set Python environment
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    # Create temp directory for Spark
    temp_dir = "C:/spark_temp"
    os.makedirs(temp_dir, exist_ok=True)

    # Build Spark session
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .config("spark.driver.memory", driver_memory)
        .config("spark.executor.memory", executor_memory)
        .config("spark.local.dir", temp_dir)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )

    print(f"[OK] Spark session created: {app_name}")
    print(f"[INFO] Spark version: {spark.version}")

    return spark


def stop_spark_session(spark):
    """Stop the Spark session"""
    if spark:
        spark.stop()
        print("[OK] Spark session stopped")


def print_dataframe_info(df, name="DataFrame"):
    """Print information about a DataFrame"""
    print("\n" + "=" * 60)
    print(f"{name} INFO")
    print("=" * 60)
    print(f"Rows: {df.count()}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumn names:")
    for col in df.columns:
        print(f"  - {col}")

    print("\nSchema:")
    df.printSchema()

    print("\nSample data:")
    df.show(5, truncate=False)


def save_as_single_csv(df, output_path):
    """Save DataFrame as a single CSV file"""
    import glob
    import shutil

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temp_dir = output_path + "_temp"

    df.coalesce(1).write \
        .mode("overwrite") \
        .option("header", True) \
        .csv(temp_dir)

    csv_files = glob.glob(f"{temp_dir}/part-*.csv")

    if csv_files:
        shutil.move(csv_files[0], output_path)
        shutil.rmtree(temp_dir)
        print(f"[OK] Saved CSV to: {output_path}")
    else:
        print("[WARNING] CSV file not found after write")


def check_null_values(df):
    """Check and report null values in DataFrame"""
    from pyspark.sql.functions import col, count, when, isnan

    print("\n[INFO] Checking for null values...")

    numeric_types = ['DoubleType', 'FloatType', 'IntegerType', 'LongType', 'DecimalType']
    numeric_cols = [
        f.name for f in df.schema.fields
        if any(dtype in str(f.dataType) for dtype in numeric_types)
    ]

    null_checks = []
    for c in df.columns:
        if c in numeric_cols:
            null_checks.append(count(when(col(c).isNull() | isnan(c), c)).alias(c))
        else:
            null_checks.append(count(when(col(c).isNull(), c)).alias(c))

    null_counts = df.select(null_checks).collect()[0].asDict()
    total_rows = df.count()

    null_cols = {k: v for k, v in null_counts.items() if v > 0}

    if null_cols:
        print(f"\n[INFO] Found null values in {len(null_cols)} columns:")
        for col_name, null_count in sorted(null_cols.items(), key=lambda x: x[1], reverse=True):
            null_pct = (null_count / total_rows) * 100
            print(f"  - {col_name:40s}: {null_count:8,} ({null_pct:6.2f}%)")
    else:
        print("[OK] No null values found")

    return null_cols


def create_directories(*paths):
    """Create multiple directories if they don't exist"""
    created = []
    for path in paths:
        if path and not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            created.append(path)

    if created:
        print(f"\n[INFO] Created {len(created)} directories:")
        for path in created:
            print(f"  - {path}")


def log_message(message, level="INFO"):
    """
    Print timestamped log message
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    level_prefix = {
        "INFO": "[INFO]",
        "WARNING": "[WARNING]",
        "ERROR": "[ERROR]",
        "SUCCESS": "[SUCCESS]",
        "START": "[START]",
        "END": "[END]"
    }

    prefix = level_prefix.get(level.upper(), "[INFO]")
    print(f"[{timestamp}] {prefix} {message}")


def print_pipeline_summary(stage, success, start_time, end_time, **kwargs):
    """
    Print summary of pipeline stage
    """
    duration = (end_time - start_time).total_seconds()

    print("\n" + "=" * 80)
    status = "COMPLETED" if success else "FAILED"
    print(f"{stage.upper()} - {status}")
    print("=" * 80)
    print(f"Start:    {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End:      {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration:.2f} seconds")

    if kwargs:
        print("\nSummary:")
        for key, value in kwargs.items():
            key_formatted = key.replace('_', ' ').title()
            if isinstance(value, (int, float)):
                print(f"  - {key_formatted}: {value}")
            else:
                print(f"  - {key_formatted}: {value}")

    print("=" * 80 + "\n")


def get_numeric_columns(df):
    """Get list of numeric columns from DataFrame"""
    numeric_types = ['DoubleType', 'FloatType', 'IntegerType', 'LongType', 'DecimalType']
    return [
        f.name for f in df.schema.fields
        if any(dtype in str(f.dataType) for dtype in numeric_types)
    ]


def print_statistics(df, columns=None, max_cols=10):
    """Print statistics for numeric columns"""
    if columns is None:
        columns = get_numeric_columns(df)

    if not columns:
        print("\n[INFO] No numeric columns found")
        return

    display_cols = columns[:max_cols]

    print(f"\nStatistics for {len(display_cols)} numeric columns:")
    if len(columns) > max_cols:
        print(f"(Showing first {max_cols} of {len(columns)} columns)")

    df.select(display_cols).describe().show()


def calculate_file_size(path):
    """Calculate total size of files in a directory (MB)"""
    if not os.path.exists(path):
        return 0

    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 * 1024)

    total_size = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, _, filenames in os.walk(path)
        for filename in filenames
    )

    return total_size / (1024 * 1024)


def validate_input_file(file_path, min_size_mb=0.001):
    """Validate that input file exists and has minimum size"""
    if not os.path.exists(file_path):
        return False, f"File not found: {file_path}"

    file_size_mb = calculate_file_size(file_path)
    if file_size_mb < min_size_mb:
        return False, f"File too small: {file_size_mb:.2f} MB"

    return True, f"File validated: {file_size_mb:.2f} MB"
