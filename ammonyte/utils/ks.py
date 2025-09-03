#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Kolmogorov-Smirnov Test for Tipping Point Detection
===================================================

This module provides the KS test method for detecting abrupt transitions 
(tipping points) in time series data, specialized for paleoclimate analysis.

Functions
---------
KS_test : Detect tipping points using Kolmogorov-Smirnov test

"""

import numpy as np
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings('ignore')

__all__ = ['KS_test']

def KS_test(series, w_min, w_max, n_w, d_c, n_c, s_c, x_c=None):
    ''' Detect tipping points using Kolmogorov-Smirnov test
    
    Applies sliding window KS test to detect abrupt transitions using the method of Bagniewski et al. (2021).
    
    Parameters
    ----------
    series : ammonyte.Series
        Series object containing time and value data
        
    w_min : float
        Size of smallest sliding window in time units
        
    w_max : float
        Size of largest sliding window in time units
        
    n_w : int
        Number of window sizes to test
        
    d_c : float
        Cut-off threshold for KS statistic
        
    n_c : int
        Minimum sample size per window
        
    s_c : float
        Standard deviation ratio threshold
        
    x_c : float
        Change threshold. If None, auto-calculated
    
    Returns
    -------
    jumps : numpy.ndarray
        Array of detected transitions with shape (n, 2)
        Each row: [time, direction] where direction is +1 (up) or -1 (down)
        
    d_statistics : numpy.ndarray
        Array of KS D-statistics corresponding to each detected transition
        
    p_values : numpy.ndarray
        Array of p-values corresponding to each detected transition
    
    Examples
    --------
    
    Basic usage (typically called via Series.kstest):
    
    .. jupyter-execute::
    
        import ammonyte as amt
        from ammonyte.utils.ks import KS_test
        ngrip = amt.Series.from_csv('ammonyte/data/NGRIP.csv')
        transitions = KS_test(ngrip, w_min=0.12, w_max=2.5, n_w=15, d_c=0.77, n_c=3, s_c=2.0)
        print(f"Detected {len(transitions)} transitions")
    
    See Also
    --------
    Series.kstest : High-level interface to this function
    
    References
    ----------
    Bagniewski, W., et al. (2021). Automatic detection of abrupt transitions in paleoclimate records. Chaos, 31(11), 113129.
    
    '''
    
    # Input validation and preprocessing
    t = np.asarray(series.time, dtype=float).flatten()
    x = np.asarray(series.value, dtype=float).flatten()
    
    # Default x_c calculation
    if x_c is None:
        x_c = np.std(x) * 0.1 + (np.max(x) - np.min(x)) * 0.05
    
    # Remove NaN values and validate data quality
    mask = ~np.isnan(t + x)
    t = t[mask]
    x = x[mask]
    
    if len(t) < 0.8 * len(series.time):
        import warnings
        warnings.warn(f"Large fraction of NaN values removed: {len(series.time) - len(t)}/{len(series.time)} points", UserWarning)
    
    if len(t) < 10:
        return np.array([[np.nan, np.nan]]), np.array([np.nan]), np.array([np.nan])  # Insufficient data after NaN removal
    
    # For dates that are equal find the mean value of x 
    # Using stable unique to preserve order 
    def unique_stable(t):
        _, idx, inv_idx = np.unique(t, return_index=True, return_inverse=True)
        # Sort by first occurrence order to preserve stability
        sort_order = np.argsort(idx)
        tt_stable = np.unique(t)[sort_order]
        # Remap inverse indices to match new ordering
        inv_idx_remapped = np.empty_like(inv_idx)
        for i, orig_pos in enumerate(sort_order):
            inv_idx_remapped[inv_idx == orig_pos] = i
        return tt_stable, inv_idx_remapped
    
    tt, indices = unique_stable(t)
    xx = np.array([np.mean(x[indices == i]) for i in range(len(tt))])
    
    # Make sure all values have the same sign
    if np.sign(np.max(xx)) > np.sign(np.min(xx)):
        xx = xx - np.min(xx) + 0.1
    
    # Varying window size 
    if w_min == w_max or n_w == 1:
        kswindow = np.array([w_min])
        n_w = 1
    else:
        kswindow = np.array([w_min * (w_max / w_min)**(i / (n_w - 1)) for i in range(n_w)])
    
    # Initialize storage arrays
    ksstat = np.zeros((len(tt), n_w))
    kspval = np.zeros((len(tt), n_w))  # Store p-values
    change_ks = np.zeros((len(tt), n_w))
    kslen1 = np.zeros((len(tt), n_w))
    kslen2 = np.zeros((len(tt), n_w))
    ksstd1 = np.zeros((len(tt), n_w))
    ksstd2 = np.zeros((len(tt), n_w))
    
    # Run the KS test 
    for j in range(n_w):
        cc = np.zeros(len(tt))
        pp = np.zeros(len(tt))  # Store p-values for this window
        changes = np.zeros(len(tt))
        len1 = np.zeros(len(tt))
        len2 = np.zeros(len(tt))
        std1 = np.zeros(len(tt))
        std2 = np.zeros(len(tt))
        
        for i in range(len(tt)):
            # locate dates for two sliding windows
            y = tt[i]
            r1 = np.where((tt > y - kswindow[j]) & (tt <= y))[0]
            r2 = np.where((tt > y) & (tt <= y + kswindow[j]))[0]
            
            # run KS test for the two windows
            if (len(r1) > 0 and len(r2) > 0 and 
                np.min(tt) <= y - kswindow[j] and 
                np.max(tt) >= y + kswindow[j]):
                
                # Use raw D-statistic (not directional) and capture p-value
                d_stat, p_value = ks_2samp(xx[r1], xx[r2], alternative='two-sided')
                cc[i] = d_stat  # Raw D-statistic
                pp[i] = p_value  # Store p-value 
            else:
                cc[i] = 0
                pp[i] = 1.0  # No significant difference
            
            # calculate additional variables
            if len(r1) > 0 and len(r2) > 0:
                changes[i] = np.mean(xx[r1]) - np.mean(xx[r2])
                len1[i] = len(r1)
                len2[i] = len(r2)
                std1[i] = np.std(xx[r1], ddof=1) if len(r1) > 1 else 0
                std2[i] = np.std(xx[r2], ddof=1) if len(r2) > 1 else 0
            else:
                changes[i] = 0
                len1[i] = 0
                len2[i] = 0
                std1[i] = 0
                std2[i] = 0
        
        # Store results with sign 
        ksstat[:, j] = cc * np.sign(changes)  # KS statistic (D_KS) + sign
        kspval[:, j] = pp  # Store p-values
        change_ks[:, j] = changes
        kslen1[:, j] = len1
        kslen2[:, j] = len2
        ksstd1[:, j] = std1
        ksstd2[:, j] = std2
    
    # Find the jumps 
    
    # Scale D_KS by mean D_KS of smaller windows
    ksstat2 = ksstat.copy()
    if ksstat.shape[1] > 2:
        for i in range(ksstat.shape[1] - 1, 2, -1):  # from last to 3rd column
            ksstat2[:, i] = np.nanmean([ksstat[:, i], np.nanmean(ksstat[:, :i], axis=1)], axis=0)
        ksstat2[:, 1] = np.nanmean([ksstat[:, 1], ksstat[:, 0]], axis=0)
    
    # Apply threshold parameters
    th_si = (kslen1 >= n_c) & (kslen2 >= n_c)
    th_st = (s_c * ksstd1 <= np.abs(change_ks)) & (s_c * ksstd2 <= np.abs(change_ks))
    th_change = x_c <= np.abs(change_ks)
    
    # Normalization formula 
    with np.errstate(divide='ignore', invalid='ignore'):
        denominator = 1 - np.sqrt((kslen1 + kslen2) / (kslen1 * kslen2))
    
    # Handle division by zero
    denominator = np.where((kslen1 == 0) | (kslen2 == 0) | (denominator == 0), np.nan, denominator)
    
    ksstat2 = 1 - (1 - np.abs(ksstat2)) / denominator
    ksstat2[np.abs(ksstat) == 1] = 1
    ksstat2[ksstat2 < 0] = 0
    ksstat2 = ksstat2 * np.sign(change_ks)
    ksstat2[th_si == 0] = 0
    ksstat2[[0, -2], :] = 0 
    
    # Replace NaN with 0
    ksstat2 = np.nan_to_num(ksstat2, nan=0.0)
    
    th_ks = np.abs(ksstat2) >= d_c
    ks_all = ksstat2 * th_st * th_si * th_change * th_ks
    
    # Append mean of all windows
    ksstat_m = ksstat.copy()
    if ks_all.shape[1] > 1:
        ksstat_m = np.column_stack([ksstat_m, np.mean(ksstat_m, axis=1)])
        ksstat2 = np.column_stack([ksstat2, np.mean(ksstat2, axis=1)])
        ks_all = np.column_stack([ks_all, np.mean(ks_all, axis=1)])
        ks_all[np.abs(ks_all) < d_c] = 0
    
    # Find D_KS peaks 
    ks_peak = []
    
    for j in range(ksstat2.shape[1]):
        k = 0
        peaks = []
        
        for i in range(1, ksstat2.shape[0]):
            if np.sign(ksstat_m[i-1, j]) != np.sign(ksstat_m[i, j]):
                segment = ks_all[k:i, j]
                if np.any(segment != 0) and not np.all(np.isnan(segment)):
                    mmm = segment.copy()
                    I = np.argmax(np.abs(mmm))
                    iii = np.where(mmm == mmm[I])[0] + k
                    
                    # Choose peak with strongest overall KS signal
                    if len(iii) > 1:
                        summed_vals = np.abs(np.sum(ksstat2[iii, :], axis=1))
                        II = np.argmax(summed_vals)
                        peak_idx = iii[II]
                    else:
                        peak_idx = iii[0]
                    
                    peaks.append([peak_idx, ks_all[peak_idx, j]])
                k = i
        
        ks_peak.append(np.array(peaks) if peaks else np.empty((0, 2)))
    
    # Check if no peaks found
    if not any(len(p) > 0 for p in ks_peak):
        return np.array([[np.nan, np.nan]]), np.array([np.nan]), np.array([np.nan])
    
    # Separate positive and negative peaks
    peak_up = []
    peak_do = []
    
    for i in range(len(ks_peak)):
        if len(ks_peak[i]) > 0:
            # Positive peaks
            pos_mask = ks_peak[i][:, 1] >= 0
            peak_up.append(ks_peak[i][pos_mask] if np.any(pos_mask) else np.empty((0, 2)))
            
            # Negative peaks
            neg_mask = ks_peak[i][:, 1] < 0
            peak_do.append(ks_peak[i][neg_mask] if np.any(neg_mask) else np.empty((0, 2)))
        else:
            peak_up.append(np.empty((0, 2)))
            peak_do.append(np.empty((0, 2)))
    
    # Initialize jump arrays
    jump_u = []
    jump_d = []
    
    # Start with largest window (last column)
    if len(peak_up) > 0 and len(peak_up[-1]) > 0:
        indices_u = peak_up[-1][:, 0].astype(int)
        jump_u = [(tt[idx] + tt[idx + 1]) / 2 for idx in indices_u if idx + 1 < len(tt)]
    
    if len(peak_do) > 0 and len(peak_do[-1]) > 0:
        indices_d = peak_do[-1][:, 0].astype(int)
        jump_d = [(tt[idx] + tt[idx + 1]) / 2 for idx in indices_d if idx + 1 < len(tt)]
    
    # Process smaller windows
    if len(ks_peak) > 1:
        for j in range(len(ks_peak) - 2, -1, -1):  # from second-to-last to first
            # Process upward peaks
            if len(peak_up[j]) > 0:
                indices_u = peak_up[j][:, 0].astype(int)
                xxx_u = [(tt[idx] + tt[idx + 1]) / 2 for idx in indices_u if idx + 1 < len(tt)]
            else:
                xxx_u = []
            
            # Process downward peaks
            if len(peak_do[j]) > 0:
                indices_d = peak_do[j][:, 0].astype(int)
                xxx_d = [(tt[idx] + tt[idx + 1]) / 2 for idx in indices_d if idx + 1 < len(tt)]
            else:
                xxx_d = []
            
            # Remove peaks close to already detected jumps
            for jump in jump_u:
                xxx_u = [x for x in xxx_u if not (jump - kswindow[j] < x < jump + kswindow[j])]
            
            for jump in jump_d:
                xxx_d = [x for x in xxx_d if not (jump - kswindow[j] < x < jump + kswindow[j])]
            
            # Append new jumps
            jump_u.extend(xxx_u)
            jump_d.extend(xxx_d)
    
    # Create final output with statistics
    jumps = []
    jump_d_stats = []
    jump_p_vals = []
    
    # For each detected jump, find the corresponding statistics
    def find_stats_for_jump(jump_time, jump_direction):
        # Find the closest time point in tt to the jump
        time_idx = np.argmin(np.abs(tt - jump_time))
        
        # Get the statistics from the mean column (last column if multiple windows)
        if ksstat.shape[1] > 1:
            d_stat = np.abs(ksstat[time_idx, -1])  # Use mean column, take absolute value
            p_val = kspval[time_idx, -1]  # Use mean column
        else:
            d_stat = np.abs(ksstat[time_idx, 0])  # Single window
            p_val = kspval[time_idx, 0]
            
        return d_stat, p_val
    
    if jump_u:
        for ju in jump_u:
            d_stat, p_val = find_stats_for_jump(ju, 1)
            jumps.append([ju, 1])
            jump_d_stats.append(d_stat)
            jump_p_vals.append(p_val)
            
    if jump_d:
        for jd in jump_d:
            d_stat, p_val = find_stats_for_jump(jd, -1)
            jumps.append([jd, -1])
            jump_d_stats.append(d_stat)
            jump_p_vals.append(p_val)
    
    if not jumps:
        return np.array([[np.nan, np.nan]]), np.array([np.nan]), np.array([np.nan])
    
    # Sort by jump time and reorder statistics accordingly
    jumps = np.array(jumps)
    jump_d_stats = np.array(jump_d_stats)
    jump_p_vals = np.array(jump_p_vals)
    
    sort_indices = np.argsort(jumps[:, 0])
    jumps = jumps[sort_indices]
    jump_d_stats = jump_d_stats[sort_indices]
    jump_p_vals = jump_p_vals[sort_indices]
    
    return jumps, jump_d_stats, jump_p_vals