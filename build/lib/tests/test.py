import unittest
import numpy as np
from Spider.main_class import SPIDER


class TestWeakAmplification(unittest.TestCase):
    def test_weak_amplification(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])
        flags = np.array([0, 1, 1, 0])
        expected_X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [3, 4], [5, 6]])
        expected_y = np.array([0, 1, 0, 1, 1, 0])
        weak_amplfication = SPIDER('weak_amplification')
        new_X, new_y, _ = weak_amplfication(X, y, flags)

        np.testing.assert_array_equal(new_X, expected_X)
        np.testing.assert_array_equal(new_y, expected_y)

    def test_weak_amplification_no_noisy_examples(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])
        flags = np.array([0, 0, 0, 0])
        expected_X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        expected_y = np.array([0, 1, 0, 1])
        weak_amplfication = SPIDER('weak_amplification')
        new_X, new_y, _ = weak_amplification(X, y, flags)

        np.testing.assert_array_equal(new_X, expected_X)
        np.testing.assert_array_equal(new_y, expected_y)

    def test_weak_amplification_and_relabeling(self):
        # Test data
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 0, 1, 1])
        flags = np.array([0, 1, 1, 0])
        
        # Expected output
        expected_X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [3, 4], [5, 6]])
        expected_y = np.array([0, 1, 1, 1, 0, 1])
        
        # Call the function to get the actual output
        weak_amplification_and_relabeling = SPIDER('weak_amplification_with_relabeling')
        actual_X, actual_y, _ = weak_amplification_and_relabeling(X, y, flags)
        
        # Compare the actual and expected outputs
        np.testing.assert_array_equal(actual_X, expected_X)
        np.testing.assert_array_equal(actual_y, expected_y)

    def test_strong_amplification(self):
        # Test data
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([0, 0, 1, 1, 1])
        flags = np.array([0, 1, 1, 0, 1])

        # Expected output
        expected_X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [3, 4], [5, 6]])
        expected_y = np.array([0, 1, 1, 0, 1, 0, 0])

        # Call the function to get the actual output
        strong_amplification = SPIDER('strong_amplification')
        actual_X, actual_y, _ = strong_amplification(X, y, flags)

        # Compare the actual and expected outputs
        np.testing.assert_array_equal(actual_X, expected_X)
        np.testing.assert_array_equal(actual_y, expected_y)

unittest.main(argv=[''], exit=False)



