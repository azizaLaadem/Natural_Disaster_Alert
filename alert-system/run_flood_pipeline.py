"""
Master Pipeline Script for Flood Prediction
Run this from the project root directory
"""
import sys
import os
import time
import subprocess

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

def run_script(script_path, step_name):
    """Run a Python script and return success status"""
    print(f"\n{'='*70}")
    print(f"EXECUTING: {step_name}")
    print(f"{'='*70}")
    print(f"Script: {script_path}\n")
    
    start_time = time.time()
    
    try:
        # Run the script as a subprocess
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=PROJECT_ROOT,
            capture_output=False,
            text=True
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n✓ {step_name} completed in {elapsed:.2f} seconds")
            return True, elapsed
        else:
            print(f"\n❌ {step_name} failed with return code {result.returncode}")
            return False, elapsed
            
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Error running {step_name}: {str(e)}")
        return False, elapsed

def main():
    print("="*70)
    print("FLOOD PREDICTION - COMPLETE ML PIPELINE")
    print("="*70)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print("\nThis pipeline will execute the following steps:")
    print("  1. Batch Ingestion (from ingestion/batch/)")
    print("  2. Data Cleaning (from processing/)")
    print("  3. Feature Engineering (from processing/)")
    print("  4. Model Training (from ml/train/)")
    print("\n" + "="*70)
    
    # Check if raw data exists
    raw_data_path = os.path.join(PROJECT_ROOT, "data/raw/flood.csv")
    if not os.path.exists(raw_data_path):
        print(f"\n❌ Error: Raw data file not found at {raw_data_path}")
        print("Please ensure the data file exists before running the pipeline.")
        return
    
    input("\nPress Enter to start the pipeline...")
    
    # Define pipeline steps
    steps = [
        ("Step 1: Batch Ingestion", os.path.join(PROJECT_ROOT, "ingestion/batch/batch_ingestion.py")),
        ("Step 2: Data Cleaning", os.path.join(PROJECT_ROOT, "processing/cleaning/clean.py")),
        ("Step 3: Feature Engineering", os.path.join(PROJECT_ROOT, "processing/feature_engineering/engineer.py")),
        ("Step 4: Model Training", os.path.join(PROJECT_ROOT, "ml/train/flood_train.py"))
    ]
    
    # Track results
    results = []
    total_time = 0
    
    # Execute each step
    for step_name, script_path in steps:
        if not os.path.exists(script_path):
            print(f"\n❌ Error: Script not found: {script_path}")
            results.append((step_name, False, 0))
            break
        
        success, elapsed = run_script(script_path, step_name)
        results.append((step_name, success, elapsed))
        total_time += elapsed
        
        if not success:
            print(f"\n❌ Pipeline failed at: {step_name}")
            print("Please check the error messages above and fix the issue.")
            break
        
        # Small pause between steps
        time.sleep(2)
    
    # Print summary
    print("\n" + "="*70)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*70)
    
    print(f"\n{'Step':<40} {'Status':<15} {'Time (s)':<10}")
    print("-"*70)
    
    all_success = True
    for step_name, success, elapsed in results:
        status = "✓ SUCCESS" if success else "❌ FAILED"
        print(f"{step_name:<40} {status:<15} {elapsed:<10.2f}")
        if not success:
            all_success = False
    
    print("-"*70)
    print(f"{'TOTAL':<40} {'':<15} {total_time:<10.2f}")
    
    if all_success:
        print("\n" + "="*70)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY! 🎉")
        print("="*70)
        print("\nYour flood prediction model is ready!")
        print(f"  - Model location: ml/models/flood_model")
        print(f"  - Metadata: ml/models/flood_metadata.json")
        print(f"  - Processed data: data/processed/flood_gold.csv")
        print("\nNext steps:")
        print("  1. Review the model metrics in the metadata file")
        print("  2. Test predictions with new data")
        print("  3. Deploy the model to production")
        print("  4. Integrate with streaming pipeline")
    else:
        print("\n" + "="*70)
        print("❌ PIPELINE FAILED")
        print("="*70)
        print("\nPlease review the error messages and fix the issues.")

if __name__ == "__main__":
    main()