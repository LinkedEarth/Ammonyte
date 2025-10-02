#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Transition Detection and Climate Analysis Results Classes
========================================================

This module contains classes to store results from various transition detection methods
and paleoclimate analysis.

Classes
-------
DeterministicTransitions : Results from deterministic transition detection methods
ClimatePhases : Results from climate phase boundary detection methods

'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
from tabulate import tabulate
import warnings
from pyleoclim.core.corr import pval_format


class DeterministicTransitions:
    ''' Container for deterministic transition detection results
    
    Stores results from tipping point detection algorithms with methods for analysis and visualization.
    
    Parameters
    ----------
    series : ammonyte.Series
        Original time series object
        
    jump_times : array-like
        Array of detected transition times
        
    jump_values : array-like  
        Array of transition directions (+1 upward, -1 downward)
        
    method : str
        Name of detection method used
        
    method_args : dict
        Dictionary of method parameters
        
    label : str
        Label for the results
        
    d_statistics : array-like, optional
        Array of KS D-statistics corresponding to each transition
        
    p_values : array-like, optional  
        Array of p-values corresponding to each transition
    
    Examples
    --------
    
    Basic transition detection:
    
    .. jupyter-execute::
    
        import ammonyte as amt
        ngrip = amt.Series.from_csv('ammonyte/data/NGRIP.csv')
        transitions = ngrip.kstest(w_min=0.12, w_max=2.5, n_w=15, d_c=0.77, n_c=3, s_c=2, x_c=0.8)
        print(transitions)
        
    Plot the results:
    
    .. jupyter-execute::
    
        transitions.plot()
        
    Access transition data:
    
    .. jupyter-execute::
    
        print(f"Number of transitions: {len(transitions.jump_times)}")
        print(f"First transition: {transitions.jump_times[0]:.2f}")
    
    See Also
    --------
    Series.kstest : Detect transitions
    ClimatePhases : Climate phase boundaries
    
    '''
    
    def __init__(self, series, jump_times, jump_values, method, method_args=None, label=None, 
                 statistics=None):
        self.series = series
        self.jump_times = np.asarray(jump_times)
        self.jump_values = np.asarray(jump_values)
        self.method = method
        self.method_args = method_args if method_args is not None else {}
        self.label = label
        self.statistics = statistics if statistics is not None else {}
        
        # Create direct attribute access for all statistics (current and future methods)
        for stat_name, stat_values in self.statistics.items():
            if stat_values is not None and len(np.asarray(stat_values)) > 0:
                setattr(self, stat_name, np.asarray(stat_values))
        
    def copy(self):
        '''
        Create a copy of the DeterministicTransitions object.
        
        Returns
        -------
        DeterministicTransitions
            A deep copy of the current object.
        '''
        return deepcopy(self)
    
    def __str__(self):
        '''
        Print summary of transition detection results.
        
        Returns
        -------
        str
            Formatted string representation of the results.
        '''
        n_jumps = len(self.jump_times)
        n_up = np.sum(self.jump_values == 1)
        n_down = np.sum(self.jump_values == -1)
        
        # Handle NaN case (no transitions detected)
        if n_jumps == 1 and np.isnan(self.jump_times[0]):
            n_jumps = 0
            n_up = 0 
            n_down = 0
        
        table = {
            'Method': [self.method],
            'Total Transitions': [n_jumps],
            'Upward Jumps': [n_up], 
            'Downward Jumps': [n_down],
            'Series Label': [getattr(self.series, 'label', 'Unlabeled')]
        }
        
        header = f"Deterministic Transition Detection Results"
        if self.label:
            display_label = self.label
        else:
            series_name = getattr(self.series,'label','Unnamed Series')
            method_desc = self.method
            
            #Add parameters info for more description
            if self.method_args:
                if 'w_min' in self.method_args and 'w_max' in self.method_args:
                    w_info = f"w={self.method_args['w_min']}-{self.method_args['w_max']}"
                    method_desc += f" ({w_info})"
                elif 'window' in self.method_args:
                    method_desc += f" (d_c={self.method_args['d_c']})"
            display_label = f"{series_name} | {method_desc}"
        
        header += f" - {display_label}"
        
        print(header)
        print("=" * len(header))
        print(tabulate(table, headers='keys', tablefmt='grid'))
        
        if n_jumps > 0 and not (n_jumps == 1 and np.isnan(self.jump_times[0])):
            print(f"\nTransition Details:")
            for i, (time, direction) in enumerate(zip(self.jump_times, self.jump_values)):
                direction_str = "Upward" if direction == 1 else "Downward"
                
                # Get time unit from original series
                time_unit = getattr(self.series, 'time_unit', None)
                if time_unit:
                    time_str = f"{time:.2f} {time_unit}"
                else:
                    time_str = f"{time:.2f}"
                
                # Add all available statistics with consistent scientific precision
                details = f"  {i+1}. Time: {time_str}, Direction: {direction_str}"
                
                for stat_name, stat_values in self.statistics.items():
                    if len(stat_values) > i:
                        if stat_name == 'p_values':
                            formatted_value = pval_format(stat_values[i])
                        else:
                            formatted_value = f"{stat_values[i]:.4f}"
                        details += f", {stat_name}: {formatted_value}"
                print(details)
        
        if self.method_args:
            print(f"\nMethod Parameters:")
            for key, value in self.method_args.items():
                print(f"  {key}: {value}")
                
        return ""
    
    def plot(self, figsize=(12, 8), ylabel=None, xlabel=None, title=None,
             upward_color='red', downward_color='blue', show_transitions='both', ax=None, **kwargs):
        '''
        Plot the time series with detected transitions.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size (width, height) in inches. Default is (12, 8).
        ylabel : str, optional
            Y-axis label. If None, uses series metadata.
        xlabel : str, optional
            X-axis label. If None, uses series metadata.
        title : str, optional
            Plot title. If None, auto-generated.
        upward_color : str, optional
            Color for upward transition markers. Default is 'red'.
        downward_color : str, optional
            Color for downward transition markers. Default is 'blue'.
        show_transitions : str, optional
            Which transitions to show: 'both', 'upward', or 'downward'. Default is 'both'.
        **kwargs
            Additional arguments passed to matplotlib plot functions.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes object.

        Examples
        --------
        >>> # Basic plot
        >>> result.plot()
        >>>
        >>> # Custom colors and size
        >>> result.plot(figsize=(15, 10), upward_color='green', downward_color='purple')
        >>>
        >>> # Show only upward transitions
        >>> result.plot(show_transitions='upward')
        '''
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Plot the original time series
        ax.plot(self.series.time, self.series.value, 'k-', linewidth=1, alpha=0.7, zorder=1, **kwargs)
        
        # Set labels
        if ylabel is None:
            ylabel = getattr(self.series, 'value_name', 'Value')
            value_unit = getattr(self.series, 'value_unit', None)
            if value_unit:
                ylabel += f' [{value_unit}]'
                
        if xlabel is None:
            xlabel = getattr(self.series, 'time_name', 'Time')
            time_unit = getattr(self.series, 'time_unit', None)
            if time_unit:
                xlabel += f' [{time_unit}]'
                
        if title is None:
            title = f'Transition Detection'
            if hasattr(self.series, 'label') and self.series.label:
                title += f' - {self.series.label}'
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        # Validate show_transitions parameter
        if show_transitions not in ['both', 'upward', 'downward']:
            raise ValueError(f"show_transitions must be 'both', 'upward', or 'downward', got '{show_transitions}'")

        # Plot transitions if any detected
        if len(self.jump_times) > 0 and not (len(self.jump_times) == 1 and np.isnan(self.jump_times[0])):
            # Separate upward and downward jumps
            upward_mask = self.jump_values == 1
            downward_mask = self.jump_values == -1

            upward_times = self.jump_times[upward_mask]
            downward_times = self.jump_times[downward_mask]

            # Plot vertical lines for transitions based on show_transitions parameter
            if show_transitions in ['both', 'upward']:
                for time in upward_times:
                    ax.axvline(time, color=upward_color, linestyle='-', linewidth=2, alpha=0.8)

            if show_transitions in ['both', 'downward']:
                for time in downward_times:
                    ax.axvline(time, color=downward_color, linestyle='-', linewidth=2, alpha=0.8)

            # Add invisible legend artists so they can be picked up by ax.legend() later
            if len(upward_times) > 0 and show_transitions in ['both', 'upward']:
                ax.plot([], [], color=upward_color, linestyle='-', linewidth=2,
                       label=f'Upward Transitions ({len(upward_times)})')
            if len(downward_times) > 0 and show_transitions in ['both', 'downward']:
                ax.plot([], [], color=downward_color, linestyle='-', linewidth=2,
                       label=f'Downward Transitions ({len(downward_times)})')

            # Create legend with all labeled artists
            ax.legend(loc='best')
        else:
            # Add text indicating no transitions found
            ax.text(0.5, 0.95, 'No transitions detected', transform=ax.transAxes, 
                   ha='center', va='top', fontsize=12, 
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig, ax
    
    def to_csv(self, path=None, **kwargs):
        '''Export detected transitions to CSV file
        
        Parameters
        ----------
        path : str, optional
            Output file path. If None, uses default naming based on method.
        **kwargs
            Additional arguments passed to pandas.to_csv()
            
        Examples
        --------
        >>> transitions = ngrip.kstest(w_min=0.12, w_max=2.5)
        >>> transitions.to_csv('my_transitions.csv')
        '''
        
        # Handle NaN case (no transitions)
        if len(self.jump_times) == 1 and np.isnan(self.jump_times[0]):
            data = {'time': [], 'time_unit': [], 'direction': [], 'jump_type': [], 
                    'd_statistic': [], 'p_value': []}
        else:
            jump_types = ['upward_transition' if d == 1 else 'downward_transition' 
                         for d in self.jump_values]
            
            # Get time unit from series
            time_unit = getattr(self.series, 'time_unit', '')
            
            # Base columns
            data = {
                'time': self.jump_times,
                'time_unit': [time_unit] * len(self.jump_times),
                'direction': self.jump_values, 
                'jump_type': jump_types,
                'method': [self.method] * len(self.jump_times)
            }
            
            # Add all available statistics
            for stat_name, stat_values in self.statistics.items():
                data[stat_name] = stat_values
        
        df = pd.DataFrame(data)
        
        if path is None:
            path = f"{self.method.replace(' ', '_')}_transitions.csv"
        
        df.to_csv(path, index=False, **kwargs)
        print(f"Transitions exported to: {path}")
        return path


class ClimatePhases:
    ''' Container for climate phase boundary detection results
    
    Stores adaptive thresholds for classifying stadial/interstadial climate phases using sliding window statistics.
    
    Parameters
    ----------
    series : ammonyte.Series
        Original time series object
        
    interstadial_bounds : array-like
        Array of warm phase thresholds (75th percentile above local mean)
        
    stadial_bounds : array-like
        Array of cold phase thresholds (25th percentile below local mean)
        
    method_args : dict
        Dictionary of algorithm parameters
    
    Examples
    --------
    
    Basic climate phase detection:
    
    .. jupyter-execute::
    
        import ammonyte as amt
        ngrip = amt.Series.from_csv('ammonyte/data/NGRIP.csv')
        phases = ngrip.find_phases(window=1.0)
        print(phases)
    
    Plot climate phases:
    
    .. jupyter-execute::
    
        phases.plot()
    
    Extract phase periods:
    
    .. jupyter-execute::
    
        periods = phases.get_periods()
        print(f"Interstadial periods: {len(periods['interstadial'])}")
        print(f"Stadial periods: {len(periods['stadial'])}")
    
    See Also
    --------
    kstest : Detect abrupt transitions
    
    References
    ----------
    Dansgaard, W., et al. (1993). Evidence for general instability of past climate from a 250-kyr ice-core record. Nature, 364(6434), 218-220.
    
    '''
    
    def __init__(self, series, interstadial_bounds, stadial_bounds, method_args=None, label=None):
        self.series = series
        self.interstadial_bounds = np.asarray(interstadial_bounds)
        self.stadial_bounds = np.asarray(stadial_bounds)
        self.method_args = method_args if method_args is not None else {}
        self.label = label
        
    def copy(self):
        '''
        Create a copy of the ClimatePhases object.
        
        Returns
        -------
        ClimatePhases
            A deep copy of the current object.
        '''
        return deepcopy(self)
    
    def __str__(self):
        '''
        Print summary of climate phase detection results.
        
        Returns
        -------
        str
            Formatted string representation of the results.
        '''
        # Validate array lengths match
        proxy_values = self.series.value
        if len(proxy_values) != len(self.interstadial_bounds) or len(proxy_values) != len(self.stadial_bounds):
            raise ValueError(f"Array length mismatch: proxy data has {len(proxy_values)} points, "
                           f"interstadial bounds has {len(self.interstadial_bounds)} points, "
                           f"stadial bounds has {len(self.stadial_bounds)} points")
        
        n_points = len(proxy_values)  # Use proxy data length, not threshold array length
        
        # Calculate statistics
        interstadial_range = np.max(self.interstadial_bounds) - np.min(self.interstadial_bounds)
        stadial_range = np.max(self.stadial_bounds) - np.min(self.stadial_bounds)
        
        # Count climate phases
        interstadial_periods = np.sum(proxy_values > self.interstadial_bounds)
        stadial_periods = np.sum(proxy_values < self.stadial_bounds)
        intermediate_periods = n_points - interstadial_periods - stadial_periods
        
        table = {
            'Method': ['find_phases'],
            'Total Points': [n_points],
            'Interstadial Periods': [interstadial_periods],
            'Stadial Periods': [stadial_periods],
            'Intermediate Periods': [intermediate_periods],
            'Series Label': [getattr(self.series, 'label', 'Unlabeled')]
        }
        
        header = f"Climate Phase Detection Results"
        if self.label:
            header += f" - {self.label}"
        
        print(header)
        print("=" * len(header))
        print(tabulate(table, headers='keys', tablefmt='grid'))
        
        print(f"\nThreshold Statistics:")
        print(f"  Interstadial range: {interstadial_range:.3f}")
        print(f"  Stadial range: {stadial_range:.3f}")
        
        if self.method_args:
            print(f"\nMethod Parameters:")
            for key, value in self.method_args.items():
                print(f"  {key}: {value}")
                
        return ""
    
    def plot(self, figsize=(12, 8), ylabel=None, xlabel=None, title=None, 
             interstadial_color='0.9', stadial_color='0.5', ax=None, **kwargs):
        '''
        Plot the time series with climate phase boundaries.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size (width, height) in inches. Default is (12, 8).
        ylabel : str, optional
            Y-axis label. If None, uses series metadata.
        xlabel : str, optional
            X-axis label. If None, uses series metadata.
        title : str, optional
            Plot title. If None, auto-generated.
        interstadial_color : str, optional
            Color for interstadial shaded regions. Default is '0.9' (very light gray).
        stadial_color : str, optional
            Color for stadial shaded regions. Default is '0.5' (medium gray).
        **kwargs
            Additional arguments passed to matplotlib plot functions.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes object.
        '''
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Validate array lengths match
        if (len(self.series.time) != len(self.interstadial_bounds) or 
            len(self.series.time) != len(self.stadial_bounds) or
            len(self.series.value) != len(self.interstadial_bounds)):
            raise ValueError(f"Array length mismatch for plotting: "
                           f"time has {len(self.series.time)} points, "
                           f"value has {len(self.series.value)} points, "
                           f"interstadial bounds has {len(self.interstadial_bounds)} points, "
                           f"stadial bounds has {len(self.stadial_bounds)} points")
        
        # Create shaded background regions for climate phases
        interstadial_mask = self.series.value > self.interstadial_bounds
        stadial_mask = self.series.value < self.stadial_bounds
        
        # Get periods as continuous segments
        periods = self.get_periods()
        
        # Create background shading - full height spans
        y_min, y_max = np.min(self.series.value), np.max(self.series.value)
        y_range = y_max - y_min
        y_min -= 0.1 * y_range  # Extend slightly
        y_max += 0.1 * y_range
        
        # Shade interstadial periods
        if len(periods['interstadial']) > 0:
            for time_val in periods['interstadial']:
                idx = np.where(self.series.time == time_val)[0]
                if len(idx) > 0:
                    i = idx[0]
                    if i < len(self.series.time) - 1:
                        width = self.series.time[i+1] - self.series.time[i]
                    else:
                        width = self.series.time[i] - self.series.time[i-1] if i > 0 else 1
                    ax.axvspan(time_val, time_val + width, ymin=0, ymax=1, 
                              color=interstadial_color, alpha=0.3, zorder=0)
        
        # Shade stadial periods  
        if len(periods['stadial']) > 0:
            for time_val in periods['stadial']:
                idx = np.where(self.series.time == time_val)[0]
                if len(idx) > 0:
                    i = idx[0]
                    if i < len(self.series.time) - 1:
                        width = self.series.time[i+1] - self.series.time[i]
                    else:
                        width = self.series.time[i] - self.series.time[i-1] if i > 0 else 1
                    ax.axvspan(time_val, time_val + width, ymin=0, ymax=1, 
                              color=stadial_color, alpha=0.3, zorder=0)
        
        # Plot the original time series on top
        ax.plot(self.series.time, self.series.value, 'k-', linewidth=1, alpha=0.8, label='Proxy Data', **kwargs)
        
        # Add legend labels (invisible elements for legend)
        from matplotlib.patches import Rectangle
        if len(periods['interstadial']) > 0:
            ax.add_patch(Rectangle((0, 0), 0, 0, color=interstadial_color, alpha=0.3, label='Interstadial Periods'))
        if len(periods['stadial']) > 0:
            ax.add_patch(Rectangle((0, 0), 0, 0, color=stadial_color, alpha=0.3, label='Stadial Periods'))
        
        # Set labels
        if ylabel is None:
            ylabel = getattr(self.series, 'value_name', 'Value')
            value_unit = getattr(self.series, 'value_unit', None)
            if value_unit:
                ylabel += f' [{value_unit}]'
                
        if xlabel is None:
            xlabel = getattr(self.series, 'time_name', 'Time')
            time_unit = getattr(self.series, 'time_unit', None)
            if time_unit:
                xlabel += f' [{time_unit}]'
                
        if title is None:
            title = f'find_phases Climate Phase Detection'
            if hasattr(self.series, 'label') and self.series.label:
                title += f' - {self.series.label}'
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig, ax
    
    def get_periods(self):
        '''
        Extract time values for each climate phase.
        
        Returns
        -------
        dict
            Dictionary containing arrays of time values for each phase:
            - 'interstadial': times where proxy > interstadial_bounds
            - 'stadial': times where proxy < stadial_bounds  
            - 'intermediate': times where stadial_bounds <= proxy <= interstadial_bounds
        '''
        proxy_values = self.series.value
        time_values = self.series.time
        
        # Validate array lengths match
        if len(proxy_values) != len(self.interstadial_bounds) or len(proxy_values) != len(self.stadial_bounds):
            raise ValueError(f"Array length mismatch: proxy data has {len(proxy_values)} points, "
                           f"interstadial bounds has {len(self.interstadial_bounds)} points, "
                           f"stadial bounds has {len(self.stadial_bounds)} points")
        
        interstadial_mask = proxy_values > self.interstadial_bounds
        stadial_mask = proxy_values < self.stadial_bounds
        intermediate_mask = (~interstadial_mask) & (~stadial_mask)
        
        return {
            'interstadial': time_values[interstadial_mask],
            'stadial': time_values[stadial_mask],
            'intermediate': time_values[intermediate_mask]
        }