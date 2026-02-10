"""
Flood Prediction Model Training - Random Forest Only
"""
import sys
import os
import time
import json
import shutil

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from processing.config import *
from processing.utils import create_spark_session, stop_spark_session, create_directories

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

def evaluate_model(predictions):
    """Evaluate model performance"""
    # Evaluators
    auc_eval = BinaryClassificationEvaluator(
        labelCol=FLOOD_TARGET_COLUMN,
        metricName="areaUnderROC"
    )
    
    acc_eval = MulticlassClassificationEvaluator(
        labelCol=FLOOD_TARGET_COLUMN,
        predictionCol="prediction",
        metricName="accuracy"
    )
    
    prec_eval = MulticlassClassificationEvaluator(
        labelCol=FLOOD_TARGET_COLUMN,
        predictionCol="prediction",
        metricName="weightedPrecision"
    )
    
    rec_eval = MulticlassClassificationEvaluator(
        labelCol=FLOOD_TARGET_COLUMN,
        predictionCol="prediction",
        metricName="weightedRecall"
    )
    
    f1_eval = MulticlassClassificationEvaluator(
        labelCol=FLOOD_TARGET_COLUMN,
        predictionCol="prediction",
        metricName="f1"
    )
    
    # Calculate metrics
    metrics = {
        'auc': auc_eval.evaluate(predictions),
        'accuracy': acc_eval.evaluate(predictions),
        'precision': prec_eval.evaluate(predictions),
        'recall': rec_eval.evaluate(predictions),
        'f1': f1_eval.evaluate(predictions)
    }
    
    return metrics

def save_model_metadata(features, val_metrics, test_metrics, model_path):
    """Save model metadata to JSON"""
    metadata = {
        'model_type': 'RandomForest',
        'disaster_type': 'flood',
        'features': features,
        'feature_count': len(features),
        'model_params': FLOOD_MODEL_PARAMS["random_forest"],
        'metrics': {
            'validation': val_metrics,
            'test': test_metrics
        },
        'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model_path': model_path
    }
    
    # Create directory if needed
    os.makedirs(os.path.dirname(FLOOD_MODEL_METADATA), exist_ok=True)
    
    with open(FLOOD_MODEL_METADATA, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    return metadata

def train_flood_model():
    """Main function to train flood prediction model"""
    print("="*60)
    print("FLOOD PREDICTION MODEL TRAINING")
    print("="*60)
    
    # Create Spark session
    spark = create_spark_session(
        app_name=SPARK_CONFIG["app_name"] + " - Flood Training",
        master=SPARK_CONFIG["master"],
        driver_memory=SPARK_CONFIG["driver_memory"],
        executor_memory=SPARK_CONFIG["executor_memory"]
    )
    
    print(f"\n✓ Spark session created (version {spark.version})")
    
    try:
        # Create necessary directories
        create_directories(MODEL_BASE_DIR)
        
        # Read ML dataset
        print(f"\n{'='*60}")
        print("Loading ML dataset...")
        print(f"{'='*60}")
        print(f"Source: {FLOOD_GOLD_PATH}")
        
        df = spark.read.csv(
            FLOOD_GOLD_PATH,
            header=True,
            inferSchema=True
        )
        
        print(f"✓ Dataset loaded: {df.count()} rows")
        print(f"✓ Features: {len(FLOOD_ML_FEATURES)}")
        
        # Show class distribution
        print("\nClass Distribution:")
        df.groupBy(FLOOD_TARGET_COLUMN).count().show()
        
        # Split data
        print(f"\n{'='*60}")
        print("Splitting data...")
        print(f"{'='*60}")
        
        train_ratio = TRAIN_TEST_SPLIT["train_ratio"]
        val_ratio = TRAIN_TEST_SPLIT["val_ratio"]
        test_ratio = TRAIN_TEST_SPLIT["test_ratio"]
        seed = TRAIN_TEST_SPLIT["seed"]
        
        train_data, temp_data = df.randomSplit([train_ratio, val_ratio + test_ratio], seed=seed)
        val_data, test_data = temp_data.randomSplit([val_ratio/(val_ratio + test_ratio), 
                                                       test_ratio/(val_ratio + test_ratio)], seed=seed)
        
        print(f"✓ Training set:   {train_data.count()} rows ({train_ratio*100:.0f}%)")
        print(f"✓ Validation set: {val_data.count()} rows ({val_ratio*100:.0f}%)")
        print(f"✓ Test set:       {test_data.count()} rows ({test_ratio*100:.0f}%)")
        
        # Build pipeline
        print(f"\n{'='*60}")
        print("Building ML Pipeline...")
        print(f"{'='*60}")
        
        assembler = VectorAssembler(
            inputCols=FLOOD_ML_FEATURES,
            outputCol="features_raw",
            handleInvalid="skip"
        )
        
        scaler = StandardScaler(
            inputCol="features_raw",
            outputCol="features",
            withStd=True,
            withMean=True
        )
        
        print("✓ Pipeline stages: VectorAssembler → StandardScaler → RandomForest")
        
        # Train Random Forest
        print(f"\n{'='*60}")
        print("Training Random Forest Model...")
        print(f"{'='*60}")
        
        # Display model parameters
        rf_params = FLOOD_MODEL_PARAMS["random_forest"]
        print("\nModel Parameters:")
        for param, value in rf_params.items():
            print(f"  {param}: {value}")
        
        start_time = time.time()
        
        # Create Random Forest classifier
        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol=FLOOD_TARGET_COLUMN,
            **rf_params
        )
        
        # Create pipeline
        rf_pipeline = Pipeline(stages=[assembler, scaler, rf])
        
        # Train model
        print("\nTraining in progress...")
        rf_model = rf_pipeline.fit(train_data)
        
        training_time = time.time() - start_time
        print(f"✓ Training completed in {training_time:.2f} seconds")
        
        # Validation evaluation
        print(f"\n{'='*60}")
        print("VALIDATION SET EVALUATION")
        print(f"{'='*60}")
        
        val_predictions = rf_model.transform(val_data)
        val_metrics = evaluate_model(val_predictions)
        
        print(f"\nValidation Metrics:")
        print(f"  AUC-ROC:   {val_metrics['auc']:.4f}")
        print(f"  Accuracy:  {val_metrics['accuracy']:.4f}")
        print(f"  Precision: {val_metrics['precision']:.4f}")
        print(f"  Recall:    {val_metrics['recall']:.4f}")
        print(f"  F1-Score:  {val_metrics['f1']:.4f}")
        
        # Test set evaluation
        print(f"\n{'='*60}")
        print("TEST SET EVALUATION")
        print(f"{'='*60}")
        
        test_predictions = rf_model.transform(test_data)
        test_metrics = evaluate_model(test_predictions)
        
        print(f"\nTest Set Metrics:")
        print(f"  AUC-ROC:   {test_metrics['auc']:.4f}")
        print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall:    {test_metrics['recall']:.4f}")
        print(f"  F1-Score:  {test_metrics['f1']:.4f}")
        
        # Confusion Matrix
        print("\nConfusion Matrix (Test Set):")
        test_predictions.groupBy(FLOOD_TARGET_COLUMN, "prediction").count().orderBy(FLOOD_TARGET_COLUMN, "prediction").show()
        
        # Show sample predictions
        print("\nSample Predictions (Test Set):")
        test_predictions.select(
            FLOOD_TARGET_COLUMN, 
            "prediction", 
            "probability"
        ).show(10, truncate=False)
        
        # Feature importance
        print(f"\n{'='*60}")
        print("FEATURE IMPORTANCE")
        print(f"{'='*60}")
        
        # Get feature importances from the Random Forest model
        rf_model_stage = rf_model.stages[-1]  # Last stage is the RandomForest
        importance = rf_model_stage.featureImportances
        feature_importance = list(zip(FLOOD_ML_FEATURES, importance.toArray()))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print("\nTop Features:")
        for i, (feat, imp) in enumerate(feature_importance, 1):
            bar = "█" * int(imp * 50)
            print(f"{i:2d}. {feat:<30} {imp:.4f} {bar}")
        
        # Save model
        print(f"\n{'='*60}")
        print("SAVING MODEL")
        print(f"{'='*60}")
        
        # Remove existing model if exists
        if os.path.exists(FLOOD_MODEL_PATH):
            shutil.rmtree(FLOOD_MODEL_PATH)
            print(f"Removed existing model at: {FLOOD_MODEL_PATH}")
        
        # Save model
        rf_model.write().overwrite().save(FLOOD_MODEL_PATH)
        print(f"✓ Model saved to: {FLOOD_MODEL_PATH}")
        
        # Save metadata
        metadata = save_model_metadata(
            FLOOD_ML_FEATURES,
            val_metrics,
            test_metrics,
            FLOOD_MODEL_PATH
        )
        print(f"✓ Metadata saved to: {FLOOD_MODEL_METADATA}")
        
        # Summary
        print(f"\n{'='*60}")
        print("MODEL TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"✓ Model Type: Random Forest")
        print(f"✓ Number of Trees: {rf_params['numTrees']}")
        print(f"✓ Max Depth: {rf_params['maxDepth']}")
        print(f"✓ Training Time: {training_time:.2f} seconds")
        print(f"\n✓ Validation Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"✓ Validation AUC-ROC: {val_metrics['auc']:.4f}")
        print(f"\n✓ Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"✓ Test AUC-ROC: {test_metrics['auc']:.4f}")
        print(f"\n✓ Model saved to: {FLOOD_MODEL_PATH}")
        print(f"✓ Metadata saved to: {FLOOD_MODEL_METADATA}")
        print(f"\n✓ Ready for deployment! 🎉")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        stop_spark_session(spark)
        print("\n✓ Spark session stopped")

if __name__ == "__main__":
    success = train_flood_model()
    sys.exit(0 if success else 1)