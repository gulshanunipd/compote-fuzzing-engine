#!/usr/bin/env python3
"""
COMPOTE Test Runner

Runs all unit tests for the COMPOTE fuzzing engine with coverage reporting.
"""

import unittest
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test modules
from tests.test_parser import TestMessageParser
from tests.test_feature_extractor import TestFeatureExtractor  
from tests.test_clustering import TestContextClustering
from tests.test_integration import TestCompoteFuzzerIntegration, TestCompoteFuzzerErrorHandling


class ColoredTextTestRunner:
    """Custom test runner with colored output"""
    
    def __init__(self, verbosity=2):
        self.verbosity = verbosity
    
    def run(self, test_suite):
        """Run the test suite with colored output"""
        print("🧪 COMPOTE Unit Test Suite")
        print("=" * 50)
        
        # Count total tests
        total_tests = test_suite.countTestCases()
        print(f"📊 Running {total_tests} tests...\n")
        
        start_time = time.time()
        
        # Run tests with default runner but capture results
        runner = unittest.TextTestRunner(
            verbosity=self.verbosity,
            stream=sys.stdout,
            descriptions=True
        )
        
        result = runner.run(test_suite)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Print summary with colors
        print("\n" + "=" * 50)
        print("📈 Test Results Summary")
        print("=" * 50)
        
        if result.wasSuccessful():
            print(f"✅ All {result.testsRun} tests passed!")
        else:
            print(f"❌ {len(result.failures)} failures, {len(result.errors)} errors")
        
        print(f"⏱️ Total time: {duration:.2f} seconds")
        print(f"📊 Tests per second: {result.testsRun / duration:.2f}")
        
        if result.failures:
            print(f"\n🔴 Failures ({len(result.failures)}):")
            for test, traceback in result.failures:
                print(f"  - {test}")
        
        if result.errors:
            print(f"\n🟠 Errors ({len(result.errors)}):")
            for test, traceback in result.errors:
                print(f"  - {test}")
        
        if result.skipped:
            print(f"\n⏭️ Skipped ({len(result.skipped)}):")
            for test, reason in result.skipped:
                print(f"  - {test}: {reason}")
        
        return result


def create_test_suite():
    """Create comprehensive test suite"""
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestMessageParser,
        TestFeatureExtractor,
        TestContextClustering, 
        TestCompoteFuzzerIntegration,
        TestCompoteFuzzerErrorHandling
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    return suite


def run_specific_tests(test_pattern: str):
    """Run tests matching a specific pattern"""
    print(f"🎯 Running tests matching pattern: {test_pattern}")
    
    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern=test_pattern)
    
    runner = ColoredTextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_coverage_analysis():
    """Run tests with coverage analysis"""
    try:
        import coverage
        
        print("📊 Running tests with coverage analysis...")
        
        # Start coverage
        cov = coverage.Coverage()
        cov.start()
        
        # Run tests
        suite = create_test_suite()
        runner = unittest.TextTestRunner(verbosity=1)
        result = runner.run(suite)
        
        # Stop coverage and generate report
        cov.stop()
        cov.save()
        
        print("\n" + "=" * 50)
        print("📈 Coverage Report")
        print("=" * 50)
        
        # Generate console report
        cov.report(show_missing=True)
        
        # Generate HTML report
        try:
            cov.html_report(directory='coverage_html')
            print("\n💾 HTML coverage report generated in 'coverage_html/'")
        except Exception as e:
            print(f"⚠️ Could not generate HTML report: {e}")
        
        return result.wasSuccessful()
        
    except ImportError:
        print("⚠️ Coverage package not installed. Install with: pip install coverage")
        print("Running tests without coverage...")
        return run_all_tests()


def run_all_tests():
    """Run all tests without coverage"""
    suite = create_test_suite()
    runner = ColoredTextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_performance_tests():
    """Run performance-focused tests"""
    print("⚡ Running performance tests...")
    
    # Create suite with performance-related tests
    suite = unittest.TestSuite()
    
    # Add specific performance tests
    performance_tests = [
        'test_selective_parsing_performance',
        'test_complexity_O_M_f', 
        'test_clustering_complexity',
        'test_performance_with_large_message_set'
    ]
    
    loader = unittest.TestLoader()
    
    for test_name in performance_tests:
        # Find and add matching tests
        for test_class in [TestMessageParser, TestFeatureExtractor, 
                          TestContextClustering, TestCompoteFuzzerIntegration]:
            if hasattr(test_class, test_name):
                suite.addTest(test_class(test_name))
    
    runner = ColoredTextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def main():
    """Main test runner function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='COMPOTE Test Runner')
    parser.add_argument('--coverage', action='store_true', 
                       help='Run with coverage analysis')
    parser.add_argument('--pattern', type=str, 
                       help='Run tests matching pattern (e.g., "test_parser*")')
    parser.add_argument('--performance', action='store_true',
                       help='Run performance tests only') 
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    try:
        if args.performance:
            success = run_performance_tests()
        elif args.pattern:
            success = run_specific_tests(args.pattern)
        elif args.coverage:
            success = run_coverage_analysis()
        else:
            success = run_all_tests()
        
        if success:
            print("\n🎉 All tests completed successfully!")
            sys.exit(0)
        else:
            print("\n💥 Some tests failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test runner error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()