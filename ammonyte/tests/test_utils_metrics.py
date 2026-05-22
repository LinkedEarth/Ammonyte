''' Tests for ammonyte.utils.metrics
Naming rules:
1. class: Test{filename}{Class}{method} with appropriate camel case
2. function: test_{method}_t{test_id}

Notes on how to test:
0. Make sure [pytest](https://docs.pytest.org) has been installed: `pip install pytest`
1. execute `pytest {directory_path}` in terminal to perform all tests in all testing files inside the specified directory
2. execute `pytest {file_path}` in terminal to perform all tests in the specified file
3. execute `pytest {file_path}::{TestClass}::{test_method}` in terminal to perform a specific test class/method inside the specified file
4. after `pip install pytest-xdist`, one may execute "pytest -n 4" to test in parallel with number of workers specified by `-n`
5. for more details, see https://docs.pytest.org/en/stable/usage.html
'''

import pytest
import numpy as np
from ammonyte.utils.metrics import evaluate_detection, DetectionMetrics


class TestDetectionMetrics:
    '''Tests for DetectionMetrics class and evaluate_detection function'''

    def test_return_type_t0(self):
        '''Test evaluate_detection returns DetectionMetrics object'''
        metrics = evaluate_detection([10, 25], [10, 26], tolerance=2)
        assert isinstance(metrics, DetectionMetrics)

    def test_attribute_access_t0(self):
        '''Test that all attributes are accessible'''
        metrics = evaluate_detection([10, 25], [10, 26], tolerance=2)

        assert hasattr(metrics, 'precision')
        assert hasattr(metrics, 'recall')
        assert hasattr(metrics, 'f1_score')
        assert hasattr(metrics, 'true_positives')
        assert hasattr(metrics, 'false_positives')
        assert hasattr(metrics, 'false_negatives')
        assert hasattr(metrics, 'tolerance')

    def test_to_dict_t0(self):
        '''Test to_dict method'''
        metrics = evaluate_detection([10, 25], [10, 26], tolerance=2)
        result = metrics.to_dict()

        assert isinstance(result, dict)
        assert 'precision' in result
        assert 'tolerance' in result

    def test_perfect_detection_t0(self):
        '''Test perfect detection scenario'''
        metrics = evaluate_detection([10, 20, 30], [10, 20, 30], tolerance=0.5)

        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.true_positives == 3
        assert metrics.false_positives == 0

    def test_no_matches_t0(self):
        '''Test when no detections match ground truth'''
        metrics = evaluate_detection([100, 200], [10, 20], tolerance=1.0)

        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.true_positives == 0

    def test_counts_consistency_t0(self):
        '''Test that TP + FP = n_detected and TP + FN = n_ground_truth'''
        metrics = evaluate_detection([10, 25, 50, 75], [10, 26, 60], tolerance=2)

        assert metrics.true_positives + metrics.false_positives == metrics.n_detected
        assert metrics.true_positives + metrics.false_negatives == metrics.n_ground_truth
