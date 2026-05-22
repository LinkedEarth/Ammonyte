#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Evaluation Metrics for Transition Detection
============================================

This module provides metrics for evaluating the performance of transition
detection methods against ground truth data.

Classes
-------
DetectionMetrics : Container for evaluation metrics results

Functions
---------
evaluate_detection : Complete evaluation with precision, recall, F1 score, and counts

'''

import numpy as np
from tabulate import tabulate

__all__ = ['DetectionMetrics', 'evaluate_detection']


class DetectionMetrics:
    ''' Container for transition detection evaluation metrics

    Stores precision, recall, F1 score, and detailed counts from comparing
    detected transitions against ground truth.

    Parameters
    ----------
    precision : float
        Precision score (TP / (TP + FP))
    recall : float
        Recall score (TP / (TP + FN))
    f1_score : float
        F1 score (harmonic mean of precision and recall)
    true_positives : int
        Number of correct detections
    false_positives : int
        Number of incorrect detections
    false_negatives : int
        Number of missed ground truth events
    n_detected : int
        Total number of detections
    n_ground_truth : int
        Total number of ground truth events
    tolerance : float
        Maximum distance for matching detected to ground truth

    '''

    def __init__(self, precision, recall, f1_score, true_positives, false_positives,
                 false_negatives, n_detected, n_ground_truth, tolerance):
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
        self.true_positives = true_positives
        self.false_positives = false_positives
        self.false_negatives = false_negatives
        self.n_detected = n_detected
        self.n_ground_truth = n_ground_truth
        self.tolerance = tolerance

    def __str__(self):
        '''
        Print summary of detection evaluation metrics.

        Returns
        -------
        str
            Formatted string representation of the results.
        '''
        header = "Detection Evaluation Metrics"
        print(header)
        print("=" * len(header))
        print(f"Metrics calculated within tolerance = {self.tolerance}")

        # Performance metrics table
        perf_table = {
            'Metric': ['Precision', 'Recall', 'F1 Score'],
            'Value': [f"{self.precision:.4f}", f"{self.recall:.4f}", f"{self.f1_score:.4f}"]
        }
        print("\nPerformance Scores:")
        print(tabulate(perf_table, headers='keys', tablefmt='grid'))

        # Counts table
        counts_table = {
            'Category': ['True Positives', 'False Positives', 'False Negatives'],
            'Count': [self.true_positives, self.false_positives, self.false_negatives]
        }
        print("\nDetection Counts:")
        print(tabulate(counts_table, headers='keys', tablefmt='grid'))

        print(f"\nSummary:")
        print(f"  Detected: {self.n_detected} | Ground Truth: {self.n_ground_truth}")

        return ""

    def to_dict(self):
        '''
        Convert metrics to dictionary.

        Returns
        -------
        dict
            Dictionary containing all metrics.
        '''
        return {
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            'n_detected': self.n_detected,
            'n_ground_truth': self.n_ground_truth,
            'tolerance': self.tolerance
        }


def evaluate_detection(detected, ground_truth, tolerance=0.5):
    ''' Complete evaluation of transition detection performance

    Calculates precision, recall, F1 score, and detailed counts (true positives,
    false positives, false negatives) in a single pass to ensure consistency.
    Uses one-to-one closest matching where each ground truth point matches only
    one detected point.

    Parameters
    ----------
    detected : array-like
        Detected transition points (indices or time values)

    ground_truth : array-like
        Ground truth transition points (indices or time values)

    tolerance : float, optional
        Maximum distance for a detected transition to be considered a true positive.
        Default is 0.5

    Returns
    -------
    DetectionMetrics
        Object containing:

        - precision : float - Precision score (0 to 1)
        - recall : float - Recall score (0 to 1)
        - f1_score : float - F1 score (0 to 1)
        - true_positives : int - Number of correct detections
        - false_positives : int - Number of incorrect detections
        - false_negatives : int - Number of missed ground truth events
        - n_detected : int - Total number of detections
        - n_ground_truth : int - Total number of ground truth events
        - tolerance : float - Tolerance used for matching

    Examples
    --------

    Basic usage:

    .. jupyter-execute::

        from ammonyte.utils.metrics import evaluate_detection

        detected = [10, 25, 50, 75]
        ground_truth = [10, 26, 60]

        metrics = evaluate_detection(detected, ground_truth, tolerance=2)
        print(metrics)

    Access individual metrics:

    .. jupyter-execute::

        from ammonyte.utils.metrics import evaluate_detection

        detected = [9.8, 23.5, 45.2]
        ground_truth = [10, 25, 40, 60]

        metrics = evaluate_detection(detected, ground_truth, tolerance=1.0)

        print(f"Precision: {metrics.precision:.3f}")
        print(f"Recall: {metrics.recall:.3f}")
        print(f"F1 Score: {metrics.f1_score:.3f}")

    '''
    detected = np.asarray(detected)
    ground_truth = np.asarray(ground_truth)

    # Handle edge cases
    if len(detected) == 0 and len(ground_truth) == 0:
        return DetectionMetrics(
            precision=0,
            recall=0,
            f1_score=0,
            true_positives=0,
            false_positives=0,
            false_negatives=0,
            n_detected=0,
            n_ground_truth=0,
            tolerance=tolerance
        )

    if len(detected) == 0:
        return DetectionMetrics(
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            true_positives=0,
            false_positives=0,
            false_negatives=len(ground_truth),
            n_detected=0,
            n_ground_truth=len(ground_truth),
            tolerance=tolerance
        )

    if len(ground_truth) == 0:
        return DetectionMetrics(
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            true_positives=0,
            false_positives=len(detected),
            false_negatives=0,
            n_detected=len(detected),
            n_ground_truth=0,
            tolerance=tolerance
        )

    # Sort arrays to ensure chronological matching
    detected = np.sort(detected)
    ground_truth = np.sort(ground_truth)

    # Use one-to-one matching: each ground truth can only match once
    matched_gt = set()
    matched_det = set()

    # Match each detected point to closest unmatched ground truth
    for i, det in enumerate(detected):
        # Find closest unmatched ground truth
        best_match = None
        best_distance = np.inf

        for j, gt in enumerate(ground_truth):
            if j not in matched_gt:
                distance = np.abs(gt - det)
                if distance <= tolerance and distance < best_distance:
                    best_match = j
                    best_distance = distance

        if best_match is not None:
            matched_gt.add(best_match)
            matched_det.add(i)

    # Calculate counts
    true_positives = len(matched_det)
    false_positives = len(detected) - true_positives
    false_negatives = len(ground_truth) - len(matched_gt)

    # Calculate metrics
    prec = true_positives / len(detected)
    rec = true_positives / len(ground_truth)
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return DetectionMetrics(
        precision=prec,
        recall=rec,
        f1_score=f1,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        n_detected=len(detected),
        n_ground_truth=len(ground_truth),
        tolerance=tolerance
    )
