"""
Configuration for Natural Disaster Alert System
"""
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
RAW_DATA_PATH = os.path.join(BASE_DIR, "data/raw")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data/processed")
STORAGE_PARQUET_PATH = os.path.join(BASE_DIR, "storage/parquet")
STORAGE_DELTA_PATH = os.path.join(BASE_DIR, "storage/delta")

# ==================== FLOOD DATA PATHS ====================
FLOOD_RAW_PATH = os.path.join(BASE_DIR, "data/raw/flood.csv")
FLOOD_INGESTED_PATH = os.path.join(STORAGE_PARQUET_PATH, "flood_ingested")
FLOOD_CLEANED_PATH = os.path.join(STORAGE_PARQUET_PATH, "flood_cleaned")
FLOOD_FEATURES_PATH = os.path.join(STORAGE_PARQUET_PATH, "flood_features")
FLOOD_GOLD_PATH = os.path.join(PROCESSED_DATA_PATH, "flood_gold.csv")

# ==================== FIRES DATA PATHS ====================
FIRES_RAW_PATH = os.path.join(BASE_DIR, "data/raw/fires.csv")
FIRES_PROCESSED_DIR = os.path.join(PROCESSED_DATA_PATH, "fires")
FIRES_INGESTED_PATH = os.path.join(STORAGE_PARQUET_PATH, "fires_ingested")
FIRES_CLEANED_PATH = os.path.join(STORAGE_PARQUET_PATH, "fires_cleaned")
FIRES_FEATURES_PATH = os.path.join(STORAGE_PARQUET_PATH, "fires_features")
FIRES_GOLD_PATH = os.path.join(PROCESSED_DATA_PATH, "fires_gold.csv")

# ==================== MODEL PATHS ====================
MODEL_BASE_DIR = os.path.join(BASE_DIR, "ml/models")
FLOOD_MODEL_PATH = os.path.join(MODEL_BASE_DIR, "flood_model")
FLOOD_MODEL_METADATA = os.path.join(MODEL_BASE_DIR, "flood_metadata.json")
FIRES_MODEL_PATH = os.path.join(MODEL_BASE_DIR, "fires_model")
FIRES_MODEL_METADATA = os.path.join(MODEL_BASE_DIR, "fires_metadata.json")

# ==================== SPARK CONFIGURATION ====================
SPARK_CONFIG = {
    "app_name": "Natural_Disaster_Alert",
    "master": "local[*]",
    "driver_memory": "6g",
    "executor_memory": "4g"
}

# ==================== FLOOD CONFIGURATION ====================
# Column mappings for flood data
FLOOD_COLUMN_MAPPING = {
    "rainfall (mm)": "rainfall_mm",
    "temperature (°c)": "temperature_c",
    "humidity (%)": "humidity_pct",
    "river discharge (m³/s)": "river_discharge",
    "water level (m)": "water_level_m",
    "elevation (m)": "elevation_m",
    "population density": "population_density",
    "historical floods": "historical_floods",
    "infrastructure": "infrastructure",
    "land cover": "land_cover",
    "soil type": "soil_type",
    "flood occurred": "flood_occurred"
}

# Feature columns
FLOOD_NUMERIC_FEATURES = [
    "rainfall_mm",
    "temperature_c",
    "humidity_pct",
    "river_discharge",
    "water_level_m",
    "elevation_m",
    "population_density",
    "historical_floods",
    "infrastructure"
]

FLOOD_CATEGORICAL_FEATURES = [
    "land_cover",
    "soil_type"
]

# ML features for flood prediction
FLOOD_ML_FEATURES = [
    "rainfall_mm",
    "water_level_m",
    "river_discharge",
    "elevation_m",
    "historical_floods",
    "rain_water_interaction",
    "discharge_per_rain",
    "flood_risk_index",
    "extreme_event"
]

FLOOD_TARGET_COLUMN = "flood_occurred"

# Model parameters for flood
FLOOD_MODEL_PARAMS = {
    "random_forest": {
        "numTrees": 200,
        "maxDepth": 20,
        "minInstancesPerNode": 2,
        "maxBins": 64,
        "featureSubsetStrategy": "sqrt",
        "subsamplingRate": 0.8,
        "seed": 42
    },
    "gradient_boosting": {
        "maxIter": 100,
        "maxDepth": 10,
        "stepSize": 0.1,
        "seed": 42
    },
    "logistic_regression": {
        "maxIter": 100,
        "regParam": 0.01,
        "seed": 42
    }
}

# ==================== FIRES CONFIGURATION ====================
# Target column for fires
FIRES_TARGET_COLUMN = "frp_class"

# Fires feature groups
FIRES_ELEVATION_FEATURES = [
    "ELEV_max", "ELEV_min", "ELEV_median", "ELEV_mean",
    "ELEV_ratio", "ELEV_range", "ELEV_diff"
]

FIRES_SLOPE_FEATURES = [
    "SLP_max", "SLP_min", "SLP_median", "SLP_mean",
    "SLP_ratio", "SLP_range", "SLP_diff"
]

FIRES_VEGETATION_FEATURES = [
    "EVT_mean", "EVH_mean", "EVC_mean", 
    "CBD_mean", "CBH_mean", "CC_mean", "CH_median"
]

FIRES_WEATHER_FEATURES = [
    "TEMP_ave", "TEMP_min", "TEMP_max",
    "PRCP", "SNOW", "WDIR_ave", "WSPD_ave", "PRES_ave", "WCOMP"
]

FIRES_ENGINEERED_FEATURES = [
    "fire_weather_index", "fire_meteo_index", "dryness_index",
    "veg_temp_interaction", "lat_lon_interaction"
]

# Model parameters for fires
FIRES_MODEL_PARAMS = {
    "random_forest": {
        "numTrees": 150,
        "maxDepth": 5,
        "seed": 42
    },
    "decision_tree": {
        "maxDepth": 15,
        "seed": 42
    },
    "gradient_boosting": {
        "maxIter": 50,
        "maxDepth": 5,
        "seed": 42
    },
    "xgboost": {
        "objective": "multi:softmax",
        "num_class": 3,
        "max_depth": 5,
        "eta": 0.1,
        "seed": 42
    }
}

# FRP classification thresholds
FIRES_FRP_THRESHOLDS = {
    "low": 800,      # FRP < 800
    "medium": 1500   # 800 <= FRP < 1500, FRP >= 1500 is high
}

# ==================== TRAINING CONFIGURATION ====================
TRAIN_TEST_SPLIT = {
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "seed": 42
}

# ==================== STREAMING CONFIGURATION ====================
KAFKA_CONFIG = {
    "bootstrap_servers": "localhost:9092",
    "flood_topic": "flood_stream",
    "fires_topic": "environmental_stream",
    "earthquake_topic": "earthquake_stream",
    "alerts_topic": "disaster_alerts"
}

# ==================== LOGGING ====================
LOG_LEVEL = "INFO"
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# ==================== HELPER FUNCTIONS ====================
def print_config():
    """Print configuration summary"""
    print("="*80)
    print("NATURAL DISASTER ALERT SYSTEM - CONFIGURATION")
    print("="*80)
    
    print(f"\n📁 BASE PATHS:")
    print(f"  Base Directory:     {BASE_DIR}")
    print(f"  Raw Data:           {RAW_DATA_PATH}")
    print(f"  Processed Data:     {PROCESSED_DATA_PATH}")
    
    print(f"\n🌊 FLOOD PATHS:")
    print(f"  Raw:       {FLOOD_RAW_PATH}")
    print(f"  Ingested:  {FLOOD_INGESTED_PATH}")
    print(f"  Cleaned:   {FLOOD_CLEANED_PATH}")
    print(f"  Features:  {FLOOD_FEATURES_PATH}")
    print(f"  Model:     {FLOOD_MODEL_PATH}")
    
    print(f"\n🔥 FIRES PATHS:")
    print(f"  Raw:       {FIRES_RAW_PATH}")
    print(f"  Ingested:  {FIRES_INGESTED_PATH}")
    print(f"  Cleaned:   {FIRES_CLEANED_PATH}")
    print(f"  Model:     {FIRES_MODEL_PATH}")
    
    print(f"\n⚙️  SPARK CONFIG:")
    for key, value in SPARK_CONFIG.items():
        print(f"  {key}: {value}")
    
    print(f"\n📡 KAFKA CONFIG:")
    for key, value in KAFKA_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    print_config()