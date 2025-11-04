import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import test_audio_utils
import test_dataset
import test_unet
import test_inference
import test_integration
import test_reverb_simulation


def run_all_tests():
    """Run all test suites"""
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test modules
    suite.addTests(loader.loadTestsFromModule(test_audio_utils))
    suite.addTests(loader.loadTestsFromModule(test_dataset))
    suite.addTests(loader.loadTestsFromModule(test_unet))
    suite.addTests(loader.loadTestsFromModule(test_inference))
    suite.addTests(loader.loadTestsFromModule(test_integration))
    suite.addTests(loader.loadTestsFromModule(test_reverb_simulation))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*60)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_all_tests())

