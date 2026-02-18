#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run all SVG tests
Usage: python run_tests.py [--verbose] [--skip-gpu]

Author: L. Morató de Dalmases
Date: February 2026
"""

import os
import sys
import argparse
import unittest
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_all_tests(verbose=False, skip_gpu=False):
    """Run all test suites"""
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern="test_*.py")
    
    # Set verbosity
    verbosity = 2 if verbose else 1
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    start_time = time.time()
    result = runner.run(suite)
    elapsed = time.time() - start_time
    
    # Print summary
    print("\n" + "="*60)
    print(f"Test Suite Summary")
    print("="*60)
    print(f"Ran {result.testsRun} tests in {elapsed:.2f} seconds")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1


def main():
    parser = argparse.ArgumentParser(description='Run SVG tests')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--skip-gpu', action='store_true',
                       help='Skip GPU tests')
    args = parser.parse_args()
    
    # Set environment variable for GPU skipping
    if args.skip_gpu:
        os.environ['SVG_SKIP_GPU_TESTS'] = '1'
    
    # Run tests
    sys.exit(run_all_tests(verbose=args.verbose, skip_gpu=args.skip_gpu))


if __name__ == '__main__':
    main()