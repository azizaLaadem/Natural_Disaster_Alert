"""
Complete Fires Prediction Pipeline
Orchestrates all steps: Ingestion -> Cleaning -> Feature Engineering -> Training
Can be run from project root directory
"""
import sys
import os
from datetime import datetime
import subprocess

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = SCRIPT_DIR  # Project root

# Add project root to path
sys.path.append(BASE_DIR)

from processing.config import *
from processing.utils import log_message

def run_command(script_path, step_name):
    """
    Run a Python script and capture output
    
    Args:
        script_path (str): Path to the Python script
        step_name (str): Name of the pipeline step
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"🚀 Starting: {step_name}")
    print(f"{'='*80}")
    print(f"Script: {script_path}")
    
    start_time = datetime.now()
    
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(script_path) or BASE_DIR
        )
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        # Check if successful
        if result.returncode == 0:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            log_message(f"{step_name} completed successfully in {duration:.2f}s", "SUCCESS")
            return True
        else:
            # Print error
            if result.stderr:
                print(f"\n❌ Error output:")
                print(result.stderr)
            
            log_message(f"{step_name} failed with return code {result.returncode}", "ERROR")
            return False
            
    except Exception as e:
        log_message(f"Error running {step_name}: {str(e)}", "ERROR")
        import traceback
        traceback.print_exc()
        return False

def run_fires_pipeline(skip_ingestion=False, skip_cleaning=False, 
                       skip_features=False, skip_training=False):
    """
    Run complete fires prediction pipeline
    
    Args:
        skip_ingestion (bool): Skip ingestion step
        skip_cleaning (bool): Skip cleaning step
        skip_features (bool): Skip feature engineering step
        skip_training (bool): Skip training step
    
    Returns:
        bool: True if all steps successful
    """
    pipeline_start = datetime.now()
    
    print("="*80)
    print("🔥 FIRES PREDICTION - COMPLETE PIPELINE")
    print("="*80)
    print(f"Pipeline started at: {pipeline_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    print(f"  - Base Directory: {BASE_DIR}")
    print(f"  - Raw Data: {FIRES_RAW_PATH}")
    print(f"  - Model Output: {FIRES_MODEL_PATH}")
    
    # Track which steps were executed
    executed_steps = []
    failed_steps = []
    
    # Define pipeline steps with flexible paths
    steps = [
        {
            'name': 'Data Ingestion',
            'script': os.path.join(BASE_DIR, 'ingestion', 'batch', 'ingest_fires.py'),
            'skip': skip_ingestion,
            'required_input': FIRES_RAW_PATH,
            'output': FIRES_INGESTED_PATH
        },
        {
            'name': 'Data Cleaning',
            'script': os.path.join(BASE_DIR, 'processing', 'cleaning', 'fires_cleaning.py'),
            'skip': skip_cleaning,
            'required_input': FIRES_INGESTED_PATH,
            'output': FIRES_CLEANED_PATH
        },
        {
            'name': 'Feature Engineering',
            'script': os.path.join(BASE_DIR, 'processing', 'feature_engineering', 'engineer_fires.py'),
            'skip': skip_features,
            'required_input': FIRES_CLEANED_PATH,
            'output': FIRES_FEATURES_PATH
        },
        {
            'name': 'Model Training',
            'script': os.path.join(BASE_DIR, 'ml', 'train', 'fires_train.py'),
            'skip': skip_training,
            'required_input': FIRES_FEATURES_PATH,
            'output': FIRES_MODEL_PATH
        }
    ]
    
    # Execute pipeline steps
    for i, step in enumerate(steps, 1):
        step_name = step['name']
        
        # Check if step should be skipped
        if step['skip']:
            log_message(f"Skipping step {i}: {step_name}", "INFO")
            continue
        
        # Check if required input exists
        required_input = step['required_input']
        if required_input and not os.path.exists(required_input):
            log_message(f"Required input not found: {required_input}", "ERROR")
            log_message(f"Cannot proceed with {step_name}", "ERROR")
            failed_steps.append(step_name)
            break
        
        # Check if script exists
        script_path = step['script']
        if not os.path.exists(script_path):
            log_message(f"Script not found: {script_path}", "ERROR")
            log_message(f"Cannot proceed with {step_name}", "ERROR")
            log_message(f"Expected location: {script_path}", "INFO")
            failed_steps.append(step_name)
            break
        
        # Run the step
        log_message(f"Step {i}/4: {step_name}", "START")
        success = run_command(script_path, step_name)
        
        if success:
            executed_steps.append(step_name)
            
            # Verify output was created
            if step['output'] and os.path.exists(step['output']):
                log_message(f"Output verified: {step['output']}", "SUCCESS")
            else:
                log_message(f"Warning: Expected output not found: {step['output']}", "WARNING")
        else:
            failed_steps.append(step_name)
            log_message(f"Pipeline failed at step: {step_name}", "ERROR")
            break
    
    # Pipeline summary
    pipeline_end = datetime.now()
    pipeline_duration = (pipeline_end - pipeline_start).total_seconds()
    
    print(f"\n{'='*80}")
    print(" PIPELINE SUMMARY")
    print(f"{'='*80}")
    print(f"Started:  {pipeline_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished: {pipeline_end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {pipeline_duration:.2f} seconds ({pipeline_duration/60:.2f} minutes)")
    
    print(f"\n Execution Summary:")
    print(f"  - Total steps: 4")
    print(f"  - Executed:    {len(executed_steps)}")
    print(f"  - Failed:      {len(failed_steps)}")
    print(f"  - Skipped:     {sum(1 for s in steps if s['skip'])}")
    
    if executed_steps:
        print(f"\nCompleted steps:")
        for step in executed_steps:
            print(f"  - {step}")
    
    if failed_steps:
        print(f"\n Failed steps:")
        for step in failed_steps:
            print(f"  - {step}")
    
    # Final status
    print(f"\n{'='*80}")
    if failed_steps:
        print("PIPELINE FAILED")
        print(f"{'='*80}")
        return False
    else:
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")
        
        # Show final outputs
        print(f"\n Pipeline Outputs:")
        print(f"  - Ingested Data:  {FIRES_INGESTED_PATH}")
        print(f"  - Cleaned Data:   {FIRES_CLEANED_PATH}")
        print(f"  - Features:       {FIRES_FEATURES_PATH}")
        print(f"  - Model:          {FIRES_MODEL_PATH}")
        
        return True

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run Fires Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python run_fires_pipeline.py
  
  # Skip ingestion (data already ingested)
  python run_fires_pipeline.py --skip-ingestion
  
  # Run only training (data already prepared)
  python run_fires_pipeline.py --skip-ingestion --skip-cleaning --skip-features
  
  # Run only feature engineering and training
  python run_fires_pipeline.py --skip-ingestion --skip-cleaning
        """
    )
    
    parser.add_argument('--skip-ingestion', action='store_true',
                       help='Skip data ingestion step')
    parser.add_argument('--skip-cleaning', action='store_true',
                       help='Skip data cleaning step')
    parser.add_argument('--skip-features', action='store_true',
                       help='Skip feature engineering step')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip model training step')
    
    args = parser.parse_args()
    
    # Show banner
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║           🔥 FIRES PREDICTION ML PIPELINE 🔥                 ║
    ║                                                               ║
    ║  Natural Disaster Alert System - Fires Module                ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    print(f"Running from: {BASE_DIR}\n")
    
    # Run pipeline
    success = run_fires_pipeline(
        skip_ingestion=args.skip_ingestion,
        skip_cleaning=args.skip_cleaning,
        skip_features=args.skip_features,
        skip_training=args.skip_training
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    # Set environment variables
    os.environ['PYARROW_IGNORE_TIMEZONE'] = '1'
    
    main()