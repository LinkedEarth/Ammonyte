#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Ruptures Change Point Detection
================================

This module provides a wrapper around the ruptures library for detecting
transitions/change points in paleoclimate time series.

Functions
---------
ruptures_transition : Detect transitions using ruptures algorithms

'''

import numpy as np
import ruptures as rpt
from ..core.transitions import DeterministicTransitions

__all__ = ['ruptures_transition']

def ruptures_transition(series, algo='Pelt', cost='rbf', pen=None, n_bkps=None,
                       min_size=2, jump=1, width=None, params=None):
    ''' Detect transitions using ruptures change point detection

    Applies ruptures algorithms for offline change point detection in time series data.

    Note that series must be evenly spaced for this method.
    See interp, bin, and gkernel methods in parent class pyleoclim.Series for details.

    Parameters
    ----------
    series : ammonyte.Series
        Series object containing time and value data

    algo : str
        Search algorithm: 'Pelt', 'Dynp', 'Binseg', 'BottomUp', 'Window', 'KernelCPD'
        Default is 'Pelt'

    cost : str
        Cost function: 'l1', 'l2', 'rbf', 'normal', 'ar', 'linear', 'rank', 'mahalanobis', 'cosine', 'clinear'
        Default is 'rbf' (Radial Basis Function - good for nonlinear patterns)
        For KernelCPD, use 'linear', 'rbf', or 'cosine' (these are kernel functions)

    pen : float, optional
        Penalty parameter controlling sensitivity to changepoints (higher = fewer changepoints)

        **What is penalty?**
        The algorithm minimizes: [fit error] + [number of changepoints × pen]
        - Low penalty → more changepoints (risk: overfitting, detecting noise as transitions)
        - High penalty → fewer changepoints (risk: missing real transitions)

        **Choosing penalty is a user decision** based on your data characteristics,
        expected transition frequency, and tolerance for false positives vs. false negatives.
        There is no single "correct" penalty value.

        **Example approaches for selecting penalty:**
        1. **Fixed empirical values**: pen = 5, 10, 20, etc. (experiment to find what works)
        2. **Information criteria** (linear penalties that balance fit vs. complexity):
           - AIC (Akaike Information Criterion) - more liberal, detects more changepoints
           - BIC (Bayesian Information Criterion) - more conservative, sample-size dependent
           - mBIC (modified BIC) - even more conservative

           **Note**: The exact formula for each criterion depends on the statistical model
           and cost function used.

    n_bkps : int, optional
        Exact number of breakpoints to detect (alternative to penalty-based approach)
        Use when you know how many transitions to expect
        **Note**: Cannot be used together with 'pen' - choose one or the other

    **Stopping criterion requirements by algorithm:**
        - Pelt: requires 'pen' (penalty)
        - Dynp: requires 'n_bkps' (number of breakpoints)
        - Binseg, BottomUp, Window, KernelCPD: requires either 'pen' or 'n_bkps' (not both)

    min_size : int
        Minimum samples between change points. Default is 2

    jump : int
        Subsample (1=all data, 5=every 5th point). Default is 1

    width : int, optional
        Window size (required for Window algorithm)

    params : dict, optional
        Additional algorithm-specific parameters

    Returns
    -------
    result : DeterministicTransitions
        Object containing detected transitions:
        - jump_times: array of transition times
        - jump_values: array of directions (+1 upward, -1 downward, 0 unclear)
        - series: original Series object
        - method: 'ruptures'
        - method_args: dict with algorithm settings
        - statistics: dict with breakpoint indices

    Examples
    --------

    Basic usage (typically called via Series.ruptures):

    .. jupyter-execute::

        import ammonyte as amt
        from ammonyte.utils.ruptures_transitions import ruptures_transition
        ngrip = amt.Series.from_csv('../data/NGRIP.csv')
        transitions = ngrip.ruptures(algo='Pelt', cost='rbf', pen=5)
        print(f"Detected {len(transitions.jump_times)} transitions")

    See Also
    --------
    Series.ruptures : High-level interface to this function

    References
    ----------
    Truong, C., et al. (2020). Selective review of offline change point detection methods. Signal Processing, 167, 107299.

    '''

    # Input validation and preprocessing
    time = series.time
    signal = series.value

    # Check if data is evenly spaced
    if not series.is_evenly_spaced():
        raise ValueError('This method assumes evenly-spaced timeseries, while the input is not. Use the ".interp()", ".bin()" or ".gkernel()" methods prior to ".ruptures()".')

    # Convert to numpy array if needed
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal)

    # Ruptures expects 2D array: (n_samples, n_features)
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    n = len(signal)

    # Validate that both pen and n_bkps are not provided simultaneously
    if pen is not None and n_bkps is not None:
        raise ValueError(
            "Cannot provide both 'pen' and 'n_bkps' parameters.\n"
            "Choose one stopping criterion:\n"
            "  - Use 'pen' for penalty-based detection (e.g., pen=10)\n"
            "  - Use 'n_bkps' for fixed number of breakpoints (e.g., n_bkps=5)"
        )

    # Validate algorithm-specific parameter requirements
    if algo == 'Pelt':
        if pen is None:
            raise ValueError(
                "Pelt algorithm requires 'pen' parameter.\n"
                "Example: pen=10 or pen=20\n"
                "For guidance on choosing penalty values, see the function documentation."
            )

    elif algo == 'Dynp':
        if n_bkps is None:
            raise ValueError(
                "Dynp algorithm requires 'n_bkps' parameter (number of breakpoints).\n"
                "Example: n_bkps=10"
            )

    elif algo == 'Window':
        if pen is None and n_bkps is None:
            raise ValueError(
                "Window algorithm requires either 'pen' or 'n_bkps' parameter.\n"
                "Example: pen=10 or pen=20, or n_bkps=10\n"
                "For guidance on choosing penalty values, see the function documentation."
            )
        if width is None:
            raise ValueError(
                "Window algorithm requires 'width' parameter.\n"
                "Example: width=100"
            )

    elif algo == 'KernelCPD':
        if pen is None and n_bkps is None:
            raise ValueError(
                "KernelCPD algorithm requires either 'pen' or 'n_bkps' parameter.\n"
                "Example: pen=10 or pen=20, or n_bkps=10\n"
                "For guidance on choosing penalty values, see the function documentation."
            )
        if cost not in ['linear', 'rbf', 'cosine']:
            raise ValueError(
                f"KernelCPD only supports 'linear', 'rbf', or 'cosine' kernel, got '{cost}'.\n"
                f"Example: cost='rbf'"
            )

    elif algo in ['Binseg', 'BottomUp']:
        if pen is None and n_bkps is None:
            raise ValueError(
                f"{algo} algorithm requires either 'pen' or 'n_bkps' parameter.\n"
                f"Example: pen=10 or pen=20, or n_bkps=10\n"
                f"For guidance on choosing penalty values, see the function documentation."
            )

    # Set up algorithm parameters
    algo_params = {
        'min_size': min_size,
        'jump': jump
    }

    # Add cost function (except for KernelCPD which uses 'kernel' parameter)
    if algo == 'KernelCPD':
        algo_params['kernel'] = cost
    else:
        algo_params['model'] = cost

    # Add Window-specific parameter
    if algo == 'Window':
        algo_params['width'] = width

    # Add any custom parameters
    if params:
        algo_params.update(params)

    # Instantiate algorithm
    try:
        algo_class = getattr(rpt, algo)
    except AttributeError:
        available = ['Pelt', 'Dynp', 'Binseg', 'BottomUp', 'Window', 'KernelCPD']
        raise ValueError(
            f"Unknown algorithm '{algo}'.\n"
            f"Available algorithms: {available}"
        )

    try:
        algo_instance = algo_class(**algo_params)
    except Exception as e:
        raise ValueError(
            f"Error initializing {algo} with parameters {algo_params}.\n"
            f"Original error: {str(e)}"
        )

    # Fit the algorithm
    try:
        algo_instance.fit(signal)
    except Exception as e:
        raise RuntimeError(
            f"Error fitting {algo} algorithm to data.\n"
            f"Signal shape: {signal.shape}, Time range: [{time[0]}, {time[-1]}]\n"
            f"Original error: {str(e)}"
        )

    # Predict breakpoints
    try:
        if pen is not None:
            breakpoints = algo_instance.predict(pen=pen)
        elif n_bkps is not None:
            breakpoints = algo_instance.predict(n_bkps=n_bkps)
        else:
            # Fallback for algorithms that don't require stopping criterion
            breakpoints = algo_instance.predict()
    except Exception as e:
        raise RuntimeError(
            f"Error predicting breakpoints with {algo}.\n"
            f"Parameters: pen={pen}, n_bkps={n_bkps}\n"
            f"Original error: {str(e)}"
        )

    # Convert indices to time values
    # Ruptures returns list of indices, last index is always len(signal)
    breakpoint_indices = np.array(breakpoints[:-1]) if len(breakpoints) > 0 else np.array([])

    if len(breakpoint_indices) == 0:
        # No transitions detected - return NaN following ammonyte convention
        jump_times = np.array([np.nan])
        jump_values = np.array([0])
        breakpoint_indices = np.array([])
    else:
        # Map indices to actual time values
        jump_times = np.array([float(time[idx]) for idx in breakpoint_indices])

        # Compute transition directions using fixed window approach
        # Note: For paleoclimate data where time increases (present → past),
        # array indices 0...idx represent RECENT data, idx...end represent OLD data
        # Direction is defined as change from past to present (chronological order):
        #   +1: warming (recent > old, upward transition)
        #   -1: cooling (recent < old, downward transition)
        #
        # We use a FIXED WINDOW around each breakpoint for consistent comparison
        # across all cost functions, using mean as the direction metric.

        # Define fixed window size (based on min_size parameter)
        window_size = max(10, min_size * 2)

        jump_values = []

        for i, idx in enumerate(breakpoint_indices):
            if idx == 0 or idx >= len(signal):
                # Changepoint at data boundaries, cannot compute direction
                jump_values.append(0)
            else:
                # Define fixed window boundaries around breakpoint
                before_start = max(0, idx - window_size)
                after_end = min(len(signal), idx + window_size)

                # Respect adjacent breakpoint boundaries to avoid overlap
                if i > 0:
                    before_start = max(before_start, breakpoint_indices[i-1])
                if i < len(breakpoint_indices) - 1:
                    after_end = min(after_end, breakpoint_indices[i+1])

                # Extract fixed-size windows
                before_vals = signal[before_start:idx].flatten()
                after_vals = signal[idx:after_end].flatten()

                # Compute direction using mean for all cost functions
                # This provides consistent, interpretable direction regardless of
                # the cost function used for breakpoint detection
                if len(before_vals) > 0 and len(after_vals) > 0:
                    before_mean = np.mean(before_vals)
                    after_mean = np.mean(after_vals)

                    if before_mean > after_mean:
                        jump_values.append(1)   # Upward transition
                    elif before_mean < after_mean:
                        jump_values.append(-1)  # Downward transition
                    else:
                        jump_values.append(0)   # No change
                else:
                    # Not enough data in window
                    jump_values.append(0)

        jump_values = np.array(jump_values)

    # Create result object
    method_args = {
        'algo': algo,
        'cost': cost,
        'pen': float(pen) if pen is not None else None,
        'n_bkps': n_bkps,
        'min_size': min_size,
        'jump': jump,
        'width': width
    }

    statistics = {
        'breakpoint_indices': breakpoint_indices.tolist() if len(breakpoint_indices) > 0 else []
    }

    result = DeterministicTransitions(
        series=series,
        jump_times=jump_times,
        jump_values=jump_values,
        method='ruptures',
        method_args=method_args,
        statistics=statistics
    )

    return result
