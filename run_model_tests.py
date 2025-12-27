#!/usr/bin/env python3
"""
Model Performance Testing Runner
Executes both MRI and Voice model comparisons and generates summary reports
"""

import os
import subprocess
import sys
import time
from datetime import datetime

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'tensorflow', 'sklearn', 'librosa', 'matplotlib', 
        'seaborn', 'pandas', 'numpy', 'opencv-python', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("ERROR: Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("SUCCESS: All required packages are installed")
    return True

def check_datasets():
    """Check if datasets are available"""
    datasets = {
        'MRI Dataset': 'parkinsons_dataset',
        'Voice Dataset': 'Parkinsons_Voice'
    }
    
    available_datasets = []
    
    for name, path in datasets.items():
        if os.path.exists(path):
            # Check if it has subdirectories
            subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            if 'normal' in subdirs and 'parkinson' in subdirs:
                print(f"SUCCESS: {name} found at {path}")
                available_datasets.append(name)
            else:
                print(f"WARNING: {name} directory exists but missing 'normal' and 'parkinson' subdirectories")
        else:
            print(f"ERROR: {name} not found at {path}")
    
    return available_datasets

def run_test(script_name, description):
    """Run a test script and capture output"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the script
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False, 
                              text=True,
                              timeout=1800)  # 30 minutes timeout
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"\nSUCCESS: {description} completed successfully in {duration:.1f} seconds")
            return True
        else:
            print(f"\nERROR: {description} failed with return code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\nTIMEOUT: {description} timed out after 30 minutes")
        return False
    except Exception as e:
        print(f"\nERROR: Error running {description}: {str(e)}")
        return False

def generate_summary_report(results):
    """Generate a summary report of all test results"""
    report_content = f"""
# Model Performance Testing Summary Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Test Execution Results

"""
    
    for test_name, success in results.items():
        status = "PASSED" if success else "FAILED"
        report_content += f"- **{test_name}**: {status}\n"
    
    report_content += f"""

## Generated Files

If tests completed successfully, the following files should be available:

### MRI Model Analysis
- `mri_model_comparison_results.csv` - Detailed performance metrics
- `mri_model_performance_comparison.png` - Performance visualization
- `mri_model_confusion_matrices.png` - Confusion matrices for all models

### Voice Model Analysis  
- `voice_model_comparison_results.csv` - Detailed performance metrics
- `voice_model_performance_comparison.png` - Performance visualization
- `voice_model_confusion_matrices.png` - Confusion matrices for all models
- `voice_model_training_histories.png` - Training curves for neural networks
- `models/comparison/` - Best performing models saved

## How to Interpret Results

### Performance Metrics
- **Accuracy**: Overall correctness of predictions
- **F1 Score**: Harmonic mean of precision and recall (best overall metric)
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

### Confusion Matrix
- **TN (True Negatives)**: Correctly predicted normal cases
- **FP (False Positives)**: Incorrectly predicted as Parkinson's
- **FN (False Negatives)**: Missed Parkinson's cases (most critical)
- **TP (True Positives)**: Correctly predicted Parkinson's cases

### Model Selection Guidelines
1. **F1 Score** is typically the best single metric for imbalanced datasets
2. **Recall** is crucial for medical diagnosis (minimize false negatives)
3. **Precision** helps avoid unnecessary worry from false positives
4. Consider **computational efficiency** for real-time applications

## Next Steps

1. Review the CSV files to identify the best performing models
2. Examine confusion matrices to understand model behavior
3. Consider ensemble methods combining top performers
4. Validate results with cross-validation or external datasets
5. Implement the best models in your production application

"""
    
    # Save report
    with open('model_testing_summary.md', 'w') as f:
        f.write(report_content)
    
    print(f"\nSummary report saved to: model_testing_summary.md")

def main():
    """Main execution function"""
    print("=" * 80)
    print("PARKINSON'S DISEASE MODEL PERFORMANCE TESTING SUITE")
    print("=" * 80)
    
    # Check dependencies
    print("\n1. Checking dependencies...")
    if not check_dependencies():
        print("\nERROR: Please install missing dependencies before running tests")
        return
    
    # Check datasets
    print("\n2. Checking datasets...")
    available_datasets = check_datasets()
    
    if not available_datasets:
        print("\nERROR: No datasets found. Please ensure datasets are properly structured:")
        print("  parkinsons_dataset/")
        print("    ├── normal/")
        print("    └── parkinson/")
        print("  Parkinsons_Voice/")
        print("    ├── normal/")
        print("    └── parkinson/")
        return
    
    # Prepare test execution
    tests_to_run = []
    results = {}
    
    if 'MRI Dataset' in available_datasets:
        tests_to_run.append(('test_mri_model_comparison.py', 'MRI Model Comparison'))
    
    if 'Voice Dataset' in available_datasets:
        tests_to_run.append(('test_voice_model_comparison.py', 'Voice Model Comparison'))
    
    if not tests_to_run:
        print("\nERROR: No valid datasets found for testing")
        return
    
    print(f"\n3. Will run {len(tests_to_run)} test suite(s)")
    
    # Ask for confirmation
    response = input(f"\nProceed with testing? This may take 30-60 minutes. (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Testing cancelled by user")
        return
    
    # Run tests
    print("\n4. Starting test execution...")
    total_start_time = time.time()
    
    for script_name, description in tests_to_run:
        success = run_test(script_name, description)
        results[description] = success
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # Generate summary
    print(f"\n{'='*80}")
    print("📋 TESTING COMPLETE")
    print(f"{'='*80}")
    print(f"Total execution time: {total_duration/60:.1f} minutes")
    
    successful_tests = sum(results.values())
    total_tests = len(results)
    
    print(f"Results: {successful_tests}/{total_tests} tests passed")
    
    for test_name, success in results.items():
        status = "PASSED" if success else "FAILED"
        print(f"  {status}: {test_name}")
    
    # Generate summary report
    generate_summary_report(results)
    
    if successful_tests == total_tests:
        print(f"\nSUCCESS: All tests completed successfully!")
        print("Check the generated files for detailed performance comparisons.")
    else:
        print(f"\nWARNING: Some tests failed. Check the output above for details.")
    
    print(f"\nSummary report: model_testing_summary.md")

if __name__ == '__main__':
    main()