"""
Model Training Script for Fires Prediction
Trains Random Forest - Stable and Fast
"""

import sys
import os
from datetime import datetime

# ==================== FIX PYTHON WORKER ISSUE ====================
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Add project root to path
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)
sys.path.append(PROJECT_ROOT)

from processing.config import *
from processing.utils import *

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark import SparkConf


def train_fires_model():
    """Train Random Forest model for fires prediction"""
    start_time = datetime.now()

    print("=" * 80)
    print("FIRES MODEL TRAINING - RANDOM FOREST")
    print("=" * 80)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python executable: {sys.executable}")

    # Create temp directory for Spark
    temp_dir = "C:/spark_temp"
    os.makedirs(temp_dir, exist_ok=True)

    # Spark session - lightweight configuration
    conf = SparkConf() \
        .setAppName("FiresRFTraining") \
        .set("spark.pyspark.python", sys.executable) \
        .set("spark.pyspark.driver.python", sys.executable) \
        .set("spark.driver.memory", "8g") \
        .set("spark.executor.memory", "8g") \
        .set("spark.local.dir", temp_dir) \
        .set("spark.sql.shuffle.partitions", "4") \
        .set("spark.default.parallelism", "4")

    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    try:
        # ==================== LOAD FEATURES ====================
        print("\n" + "=" * 80)
        print("Loading engineered features")
        print("=" * 80)
        print(f"Source: {FIRES_FEATURES_PATH}")

        df = spark.read.parquet(FIRES_FEATURES_PATH)
        total_rows = df.count()
        total_cols = len(df.columns)

        print(f" Loaded {total_rows:,} rows and {total_cols} columns")

        # ==================== PREPARE DATA ====================
        print("\n" + "=" * 80)
        print("Preparing ML dataset")
        print("=" * 80)

        target = "frp_class"
        exclude_cols = ["frp", "frp_class", "Neighbour_frp"]

        features = [c for c in df.columns if c not in exclude_cols]

        print(f" Number of features: {len(features)}")
        print(f" Target column: {target}")

        print("\n Target distribution:")
        df.groupBy(target).count().orderBy(target).show()

        # Assemble features
        assembler = VectorAssembler(
            inputCols=features,
            outputCol="features",
            handleInvalid="skip"
        )

        print("\n Assembling features...")
        df_ml = assembler.transform(df).select("features", target)

        ml_count = df_ml.count()
        print(f" ML dataset ready: {ml_count:,} rows")

        # ==================== TRAIN / TEST SPLIT ====================
        print("\n" + "=" * 80)
        print("Train-Test Split")
        print("=" * 80)

        train, test = df_ml.randomSplit([0.8, 0.2], seed=42)

        train_count = train.count()
        test_count = test.count()

        print(f" Train set: {train_count:,} rows ({train_count/ml_count*100:.1f}%)")
        print(f" Test set:  {test_count:,} rows ({test_count/ml_count*100:.1f}%)")

        # ==================== TRAIN RANDOM FOREST ====================
        print("\n" + "=" * 80)
        print("Training Random Forest Classifier")
        print("=" * 80)

        rf = RandomForestClassifier(
            labelCol=target,
            featuresCol="features",
            numTrees=100,         # Reduced for stability
            maxDepth=10,          # Reduced for stability
            maxBins=32,
            seed=42
        )

        print(" Training model (this may take a few minutes)...")
        model = rf.fit(train)
        print(" Training completed")

        # ==================== EVALUATION ====================
        print("\n" + "=" * 80)
        print("Model Evaluation")
        print("=" * 80)

        print(" Generating predictions...")
        predictions = model.transform(test)

        # Evaluators
        evaluator_f1 = MulticlassClassificationEvaluator(
            labelCol=target,
            predictionCol="prediction",
            metricName="f1"
        )

        evaluator_acc = MulticlassClassificationEvaluator(
            labelCol=target,
            predictionCol="prediction",
            metricName="accuracy"
        )

        evaluator_precision = MulticlassClassificationEvaluator(
            labelCol=target,
            predictionCol="prediction",
            metricName="weightedPrecision"
        )

        evaluator_recall = MulticlassClassificationEvaluator(
            labelCol=target,
            predictionCol="prediction",
            metricName="weightedRecall"
        )

        print(" Calculating metrics...")
        
        # Calculate metrics sequentially to avoid memory issues
        acc = evaluator_acc.evaluate(predictions)
        print(f"   Accuracy calculated: {acc:.4f}")
        
        f1 = evaluator_f1.evaluate(predictions)
        print(f"   F1-Score calculated: {f1:.4f}")
        
        precision = evaluator_precision.evaluate(predictions)
        print(f"   Precision calculated: {precision:.4f}")
        
        recall = evaluator_recall.evaluate(predictions)
        print(f"   Recall calculated: {recall:.4f}")

        print("\n" + "=" * 60)
        print(" MODEL PERFORMANCE")
        print("=" * 60)
        print(f"   Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
        print(f"   F1-score:  {f1:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print("=" * 60)

        # ==================== FEATURE IMPORTANCE ====================
        print("\n" + "=" * 80)
        print("Feature Importance (Top 10)")
        print("=" * 80)

        feature_importance = list(zip(features, model.featureImportances.toArray()))
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        for i, (feat, importance) in enumerate(feature_importance[:10], 1):
            print(f"  {i:2d}. {feat:40s} : {importance:.4f}")

        # ==================== SAVE MODEL ====================
        print("\n" + "=" * 80)
        print("Saving model")
        print("=" * 80)

        model_path = os.path.join(FIRES_MODEL_PATH, "random_forest")
        create_directories(model_path)

        model.write().overwrite().save(model_path)
        print(f" Model saved to: {model_path}")

        # Save feature names
        feature_names_path = os.path.join(FIRES_MODEL_PATH, "feature_names.txt")
        with open(feature_names_path, 'w') as f:
            for feature in features:
                f.write(f"{feature}\n")
        print(f" Feature names saved")

        # Save feature importance
        importance_path = os.path.join(FIRES_MODEL_PATH, "feature_importance.txt")
        with open(importance_path, 'w') as f:
            f.write("Feature,Importance\n")
            for feat, imp in feature_importance:
                f.write(f"{feat},{imp}\n")
        print(f" Feature importance saved")

        # Save metrics
        metrics_path = os.path.join(FIRES_MODEL_PATH, "metrics.txt")
        with open(metrics_path, 'w') as f:
            f.write(f"Model: Random Forest\n")
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Samples: {ml_count}\n")
            f.write(f"Train Samples: {train_count}\n")
            f.write(f"Test Samples: {test_count}\n")
            f.write(f"Features: {len(features)}\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"F1-Score: {f1:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
        print(f"Metrics saved")

        # ==================== SUMMARY ====================
        end_time = datetime.now()

        print_pipeline_summary(
            stage="Model Training (Random Forest)",
            success=True,
            start_time=start_time,
            end_time=end_time,
            total_samples=ml_count,
            train_samples=train_count,
            test_samples=test_count,
            num_features=len(features),
            model_type="Random Forest (100 trees, depth 10)",
            accuracy=f"{acc:.4f}",
            f1_score=f"{f1:.4f}",
            precision=f"{precision:.4f}",
            recall=f"{recall:.4f}",
            model_path=model_path
        )

        return True

    except Exception as e:
        print("\n" + "=" * 80)
        print(" ERROR DURING MODEL TRAINING")
        print("=" * 80)
        print(f"Error: {str(e)}")
        print("\n Full traceback:")
        import traceback
        traceback.print_exc()
        return False

    finally:
        spark.stop()
        print("\n Spark session stopped")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

    print("\n" + "=" * 80)
    print("FIRES MODEL TRAINING PIPELINE - RANDOM FOREST")
    print("=" * 80)
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Python path: {sys.executable}")
    
    success = train_fires_model()
    
    if success:
        print("\n Training completed successfully!")
        print("\n Model files saved:")
        print(f"  - Model: {FIRES_MODEL_PATH}/random_forest")
        print(f"  - Feature names: {FIRES_MODEL_PATH}/feature_names.txt")
        print(f"  - Feature importance: {FIRES_MODEL_PATH}/feature_importance.txt")
        print(f"  - Metrics: {FIRES_MODEL_PATH}/metrics.txt")
        sys.exit(0)
    else:
        print("\n Training failed!")
        sys.exit(1)