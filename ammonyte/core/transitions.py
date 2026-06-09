#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Transition Detection Results Classes
=====================================

This module contains classes to store results from transition detection methods.

Classes
-------
DeterministicTransitions : Results from deterministic transition detection methods

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
    
        import os, ammonyte as amt
        ngrip = amt.Series.from_csv(os.path.join(os.path.dirname(amt.__file__), 'data', 'NGRIP.csv'))
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
             upward_color='red', downward_color='blue', transition_color=None,
             show_transitions='both', ax=None,
             title_fontsize=14, label_fontsize=12, tick_fontsize=10, legend_fontsize=10,
             show_legend=True, legend_labels=None, legend_loc='best', **kwargs):
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
            Used when show_transitions is 'both' or 'upward'.
        downward_color : str, optional
            Color for downward transition markers. Default is 'blue'.
            Used when show_transitions is 'both' or 'downward'.
        transition_color : str, optional
            Color for all transitions when show_transitions='all'. Default is None.
            If None, falls back to upward_color. Recommended to use this parameter
            when show_transitions='all' for clarity.
        show_transitions : str, optional
            Which transitions to show: 'all', 'both', 'upward', or 'downward'. Default is 'both'.
            - 'all': Show all transitions in one color without direction distinction (use transition_color)
            - 'both': Show upward and downward transitions in different colors (use upward_color/downward_color)
            - 'upward': Show only upward transitions (use upward_color)
            - 'downward': Show only downward transitions (use downward_color)
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, creates new figure.
        title_fontsize : int, optional
            Font size for plot title. Default is 14.
        label_fontsize : int, optional
            Font size for axis labels. Default is 12.
        tick_fontsize : int, optional
            Font size for tick labels (numbers on axes). Default is 10.
        legend_fontsize : int, optional
            Font size for legend. Default is 10.
        show_legend : bool, optional
            Whether to display legend. Default is True.
        legend_labels : list of str, optional
            Custom legend labels. If None, uses auto-generated labels.
            Provide as list: ['label1'] for single transition type, or ['label1', 'label2'] for both.
        legend_loc : str, optional
            Legend location. Default is 'best'. Options: 'upper right', 'upper left',
            'lower left', 'lower right', 'right', 'center left', 'center right',
            'lower center', 'upper center', 'center', 'best'.
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
        >>> result.plot(show_transitions='upward', upward_color='red')
        >>>
        >>> # Show all transitions without direction distinction (single color, single legend)
        >>> result.plot(show_transitions='all', transition_color='blue')
        >>>
        >>> # Custom font sizes
        >>> result.plot(title_fontsize=18, label_fontsize=16, tick_fontsize=14, legend_fontsize=14)
        >>>
        >>> # Hide legend
        >>> result.plot(show_legend=False)
        >>>
        >>> # Custom legend label with 'all' mode
        >>> result.plot(show_transitions='all', transition_color='purple',
        ...             legend_labels=['Detected Transitions'])
        >>>
        >>> # Custom legend location and color
        >>> result.plot(show_transitions='all', transition_color='#1f77b4',
        ...             legend_loc='upper right')
        '''
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Plot the original time series
        ax.plot(self.series.time, self.series.value, 'k-', linewidth=1, alpha=0.7, zorder=1, **kwargs)
        
        # Set labels
        if ylabel is None:
            ylabel = getattr(self.series, 'value_name', None) or 'Value'
            value_unit = getattr(self.series, 'value_unit', None)
            if value_unit:
                ylabel += f' [{value_unit}]'

        if xlabel is None:
            xlabel = getattr(self.series, 'time_name', None) or 'Time'
            time_unit = getattr(self.series, 'time_unit', None)
            if time_unit:
                xlabel += f' [{time_unit}]'
                
        if title is None:
            title = f'Transition Detection'
            if hasattr(self.series, 'label') and self.series.label:
                title += f' - {self.series.label}'

        ax.set_xlabel(xlabel, fontsize=label_fontsize)
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
        ax.set_title(title, fontsize=title_fontsize)
        ax.tick_params(axis='both', labelsize=tick_fontsize)

        # Validate show_transitions parameter
        if show_transitions not in ['all', 'both', 'upward', 'downward']:
            raise ValueError(f"show_transitions must be 'all', 'both', 'upward', or 'downward', got '{show_transitions}'")

        # Plot transitions if any detected
        if len(self.jump_times) > 0 and not (len(self.jump_times) == 1 and np.isnan(self.jump_times[0])):
            # Separate upward and downward jumps
            upward_mask = self.jump_values == 1
            downward_mask = self.jump_values == -1

            upward_times = self.jump_times[upward_mask]
            downward_times = self.jump_times[downward_mask]

            # Plot vertical lines for transitions based on show_transitions parameter
            if show_transitions == 'all':
                # Plot all transitions in one color (no direction distinction)
                # Use transition_color if specified, otherwise fall back to upward_color
                color = transition_color if transition_color is not None else upward_color
                for time in self.jump_times:
                    ax.axvline(time, color=color, linestyle='-', linewidth=2, alpha=0.8)
            else:
                # Plot transitions with direction distinction
                if show_transitions in ['both', 'upward']:
                    for time in upward_times:
                        ax.axvline(time, color=upward_color, linestyle='-', linewidth=2, alpha=0.8)

                if show_transitions in ['both', 'downward']:
                    for time in downward_times:
                        ax.axvline(time, color=downward_color, linestyle='-', linewidth=2, alpha=0.8)

            # Create legend if requested
            if show_legend:
                if show_transitions == 'all':
                    # Single legend entry for all transitions
                    # Use transition_color if specified, otherwise fall back to upward_color
                    color = transition_color if transition_color is not None else upward_color
                    if legend_labels is not None:
                        if not isinstance(legend_labels, (list, tuple)):
                            legend_labels = [legend_labels]
                        label = legend_labels[0] if len(legend_labels) > 0 else f'Transitions ({len(self.jump_times)})'
                    else:
                        label = f'Transitions ({len(self.jump_times)})'
                    ax.plot([], [], color=color, linestyle='-', linewidth=2, label=label)
                else:
                    # Separate legend entries for upward/downward or filtered transitions
                    if legend_labels is not None:
                        # Use custom labels
                        if not isinstance(legend_labels, (list, tuple)):
                            legend_labels = [legend_labels]

                        # Create legend entries with custom labels
                        label_idx = 0
                        if len(upward_times) > 0 and show_transitions in ['both', 'upward']:
                            label = legend_labels[label_idx] if label_idx < len(legend_labels) else f'Upward Transitions ({len(upward_times)})'
                            ax.plot([], [], color=upward_color, linestyle='-', linewidth=2, label=label)
                            label_idx += 1

                        if len(downward_times) > 0 and show_transitions in ['both', 'downward']:
                            label = legend_labels[label_idx] if label_idx < len(legend_labels) else f'Downward Transitions ({len(downward_times)})'
                            ax.plot([], [], color=downward_color, linestyle='-', linewidth=2, label=label)
                    else:
                        # Use default labels
                        if len(upward_times) > 0 and show_transitions in ['both', 'upward']:
                            ax.plot([], [], color=upward_color, linestyle='-', linewidth=2,
                                   label=f'Upward Transitions ({len(upward_times)})')
                        if len(downward_times) > 0 and show_transitions in ['both', 'downward']:
                            ax.plot([], [], color=downward_color, linestyle='-', linewidth=2,
                                   label=f'Downward Transitions ({len(downward_times)})')

                # Create legend with specified location
                ax.legend(loc=legend_loc, fontsize=legend_fontsize)
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