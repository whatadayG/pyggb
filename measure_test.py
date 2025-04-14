import os
import sys
import traceback
import numpy as np
import shutil
from collections import Counter
from random_constr import Construction

def test_measure_construction(file_path, num_tests=20, precision=4, verbose=True):
    """
    Test a geometric construction that ends with a measure statement.
    
    Args:
        file_path: Path to the construction file
        num_tests: Number of tests to run
        precision: Number of decimal places to round measurements to
        verbose: Whether to print detailed output
    
    Returns:
        A dictionary with statistics about the measurements
    """
    construction = Construction()
    try:
        construction.load(file_path)
    except Exception as e:
        if verbose:
            print(f"Error loading {file_path}: {str(e)}")
        return None
    
    if construction.statement_type != "measure":
        if verbose:
            print(f"Construction in {file_path} does not end with a measure statement")
        return None
    
    measurements = []
    failures = 0
    
    for i in range(num_tests):
        try:
            construction.run_commands()
            value = construction.to_measure.value()
            measurements.append(value)
            if verbose:
                print(f"Test {i+1}: {value}")
        except Exception as e:
            failures += 1
            if verbose:
                print(f"Test {i+1} failed: {str(e)}")
                traceback.print_exc()
    
    if not measurements:
        if verbose:
            print("All tests failed!")
        return None
    
    # Round measurements to specified precision for analysis
    rounded_measurements = [round(m, precision) for m in measurements]
    counts = Counter(rounded_measurements)
    
    # Calculate statistics
    avg = np.mean(measurements)
    median = np.median(measurements)
    min_val = min(measurements)
    max_val = max(measurements)
    
    # Most common value and its count
    most_common = counts.most_common(1)[0]
    mode = most_common[0]
    mode_count = most_common[1]
    
    results = {
        "successful_tests": len(measurements),
        "failed_tests": failures,
        "average": avg,
        "median": median,
        "min": min_val,
        "max": max_val,
        "mode": mode,
        "mode_count": mode_count,
        "all_values": measurements,
        "counts": dict(counts),
        "pass": mode_count >= 18 and len(measurements) >= 18
    }
    
    if verbose:
        print(f"\nResults Summary:")
        print(f"Tests: {len(measurements)} successful, {failures} failed")
        print(f"Average: {avg}")
        print(f"Median: {median}")
        print(f"Min: {min_val}")
        print(f"Max: {max_val}")
        print(f"Most common value: {mode} (occurs {mode_count} times out of {len(measurements)})")
        print(f"PASS: {results['pass']}")
    
    return results

def process_directory(directory_path, passed_dir="passed", failed_dir="failed", num_tests=20, precision=4, verbose=False):
    """
    Process all .txt files in a directory and sort them into pass/fail directories.
    
    Args:
        directory_path: Path to directory containing construction files
        passed_dir: Directory to move passing files to
        failed_dir: Directory to move failing files to
        num_tests: Number of tests to run for each file
        precision: Number of decimal places to round measurements to
        verbose: Whether to print detailed output for each test
    
    Returns:
        Dictionary with results for each file
    """
    # Create output directories if they don't exist
    os.makedirs(passed_dir, exist_ok=True)
    os.makedirs(failed_dir, exist_ok=True)
    
    results = {}
    passed_results = {}  # Store results of passing files for answers.txt
    
    # Process each .txt file in the directory
    for filename in os.listdir(directory_path):
        if not filename.endswith('.txt'):
            continue
            
        file_path = os.path.join(directory_path, filename)
        print(f"Testing {filename}...")
        
        # Test the construction
        test_results = test_measure_construction(file_path, num_tests, precision, verbose)
        
        if test_results is None:
            print(f"  Failed to test {filename}")
            results[filename] = {"status": "error", "passed": False}
            # Move to failed directory
            shutil.move(file_path, os.path.join(failed_dir, filename))
            continue
        
        # Determine if file passed or failed
        passed = test_results["pass"]
        results[filename] = {"status": "pass" if passed else "fail", "results": test_results}
        
        # If passed, store the most common value for answers.txt
        if passed:
            passed_results[filename] = test_results["mode"]
        
        # Move file to appropriate directory
        destination = os.path.join(passed_dir if passed else failed_dir, filename)
        shutil.move(file_path, destination)
        
        print(f"  {'PASSED' if passed else 'FAILED'}: {test_results['mode_count']} of {test_results['successful_tests']} tests gave the same result")
    
    # Create answers.txt in the passed directory with the values from passed files
    if passed_results:
        answers_path = os.path.join(passed_dir, "answers.txt")
        with open(answers_path, 'a') as f:
            for filename, value in passed_results.items():
                f.write(f"{filename}: {value}\n")
        print(f"Created answers.txt in {passed_dir} with {len(passed_results)} entries")
    
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python measure_test.py <construction_file|directory> [num_tests]")
        sys.exit(1)
    
    path = sys.argv[1]
    num_tests = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    
    if os.path.isdir(path):
        print(f"Processing directory: {path}")
        results = process_directory(path, num_tests=num_tests)
        
        # Print summary
        passed = sum(1 for r in results.values() if r.get("status") == "pass")
        failed = sum(1 for r in results.values() if r.get("status") == "fail")
        errors = sum(1 for r in results.values() if r.get("status") == "error")
        
        print(f"\nSummary: {passed} passed, {failed} failed, {errors} errors")
    else:
        test_measure_construction(path, num_tests) 