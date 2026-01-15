#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LERM Transition Detection
=========================

This module provides functions for detecting transitions using the LERM
(Laplacian Eigenmaps Recurrence Matrix) Fisher Information approach.

Functions
---------
lerm_transition : Detect upward and downward transitions from Fisher Information series

"""

import numpy as np
from ..utils.sampling import confidence_interval

__all__ = ['lerm_transition']


def lerm_transition(series, transition_interval=None,
                    upper=95, lower=5, w=50, n_samples=10000):
    ''' Detect transitions using LERM Fisher Information threshold crossings

    Identifies complete transitions where the Fisher Information series crosses
    from one confidence bound to the other, indicating abrupt changes in the
    dynamical system.

    Parameters
    ----------
    series : ammonyte.Series, ammonyte.RQARes
        Fisher Information series from LERM analysis (typically smoothed)

    transition_interval : tuple, optional
        (upper_bound, lower_bound) for detecting transitions.
        If None, automatically computes via bootstrapping using the parameters below.

    upper : int, optional
        Upper percentile for confidence interval. Default is 95
        Only used if transition_interval is None.

    lower : int, optional
        Lower percentile for confidence interval. Default is 5
        Only used if transition_interval is None.

    w : int, optional
        Bootstrap sample size. Default is 50
        Only used if transition_interval is None.

    n_samples : int, optional
        Number of bootstrap samples. Default is 10000
        Only used if transition_interval is None.

    Returns
    -------
    jump_times : numpy.ndarray
        Array of transition times

    jump_directions : numpy.ndarray
        Array of transition directions (+1 for upward, -1 for downward)

    upper_bound : float
        Upper confidence bound used for detection

    lower_bound : float
        Lower confidence bound used for detection

    Examples
    --------

    Basic usage (typically called via Series.lerm_transitions):

    .. jupyter-execute::

        import ammonyte as amt
        from ammonyte.utils.lerm_transitions import lerm_transition

        # Load data and perform LERM analysis
        ngrip = amt.Series.from_csv('../data/NGRIP.csv')
        NGRIP_td = amt.TimeEmbeddedSeries(ngrip, m=11)
        NGRIP_epsilon = NGRIP_td.find_epsilon(eps=1, target_density=0.05)
        NGRIP_lp = NGRIP_epsilon.laplacian_eigenmaps(w_size=20, w_incre=4)
        NGRIP_lp_smooth = amt.utils.fisher.smooth_series(NGRIP_lp, block_size=3)

        # Detect transitions
        jump_times, jump_directions, upper_bound, lower_bound = lerm_transition(NGRIP_lp_smooth)
        print(f"Detected {len(jump_times)} transitions")
        print(f"Bounds used: upper={upper_bound:.4f}, lower={lower_bound:.4f}")

    See Also
    --------
    Series.lerm_transitions : High-level interface to this function
    confidence_interval : Bootstrap confidence interval calculation

    Notes
    -----

    This function detects complete transitions where the series crosses from one
    confidence threshold to the other. The algorithm:

    1. Interpolates the series to step=1 for fine-grained detection
    2. Computes or uses provided confidence bounds (upper, lower)
    3. Tracks crossings above the upper threshold
    4. Tracks crossings below the lower threshold
    5. Identifies complete transitions:
       - Upward (+1): from below lower bound to above upper bound
       - Downward (-1): from above upper bound to below lower bound
    6. Assigns transition time at the midpoint between threshold crossings
    '''

    # Compute confidence bounds if not provided
    if transition_interval is None:
        upper_bound, lower_bound = confidence_interval(
            series, upper=upper, lower=lower, w=w, n_samples=n_samples
        )
    else:
        upper_bound, lower_bound = transition_interval

    # Interpolate to fine resolution for accurate transition detection
    series_fine = series.interp(step=1)

    # Detect threshold crossings
    above_thresh = np.where(series_fine.value > upper_bound, 1, 0)
    below_thresh = np.where(series_fine.value < lower_bound, 1, 0)

    transition_above = np.diff(above_thresh)
    transition_below = np.diff(below_thresh)

    # Track complete transitions and their directions
    full_trans = np.zeros(len(transition_above))
    trans_directions = np.zeros(len(transition_above))

    last_above = 0
    last_below = 0
    above_pointer = 0
    below_pointer = 0

    for i in range(len(transition_above)):
        above = transition_above[i]
        below = transition_below[i]

        if above != 0:
            if last_below + above == 0:
                # Complete transition between bounds
                loc = int((i + below_pointer) / 2)
                full_trans[loc] = 1
                # Direction is determined by the sign of 'above' crossing
                # above = +1 means upward, above = -1 means downward
                trans_directions[loc] = above
                last_below = 0
            last_above = above
            above_pointer = i

        if below != 0:
            if last_above + below == 0:
                # Complete transition between bounds
                loc = int((i + above_pointer) / 2)
                full_trans[loc] = 1
                # Direction is determined by the sign of 'below' crossing
                # below = +1 means downward, below = -1 means upward
                trans_directions[loc] = -below
                last_above = 0
            last_below = below
            below_pointer = i

    # Extract transition times and directions
    transition_mask = full_trans != 0
    jump_times = series_fine.time[1:][transition_mask]
    jump_directions = trans_directions[transition_mask].astype(int)

    # Reverse directions for backward time axes (e.g., kyr b2k)
    if np.mean(np.diff(series.time)) > 0:
        jump_directions = -jump_directions

    return jump_times, jump_directions, upper_bound, lower_bound
