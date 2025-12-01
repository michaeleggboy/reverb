import unittest
import sys
import test_audio_utils
import test_dataset
import test_unet
import test_inference
import test_integration
import test_reverb_simulation


def run_all_tests():
    """Run all test suites"""
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromModule(test_audio_utils))
    suite.addTests(loader.loadTestsFromModule(test_dataset))
    suite.addTests(loader.loadTestsFromModule(test_unet))
    suite.addTests(loader.loadTestsFromModule(test_inference))
    suite.addTests(loader.loadTestsFromModule(test_integration))
    suite.addTests(loader.loadTestsFromModule(test_reverb_simulation))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*60)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_all_tests())

