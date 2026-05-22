#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pyleoclim as pyleo
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .sampling import confidence_interval

__all__ = [
    'bootstrap_fill_plot',
    'bootstrap_scatter_plot'
]

def get_labels(obj):
    '''Create plotting labels from object metadata

    Assumes specific object properties specific to ammonyte objects.

    Parameters
    ----------

    obj : ammonyte.TimeEmbeddedSeries, ammonyte.RecurrenceMatrix, or ammonyte.RecurrenceNetwork
        Object whose metadata will be used to construct axis labels.

    Returns
    -------

    xlabel : str or None
        Label string for the x-axis, formatted as 'time_name [time_unit]' if both
        are available, or just 'time_name', or None if neither is available.

    ylabel : str or None
        Label string for the y-axis, formatted as 'value_name [value_unit]' if both
        are available, or just 'value_name', or None if neither is available.
    '''

    if obj.time_name is not None:
        if obj.time_unit is not None:
            xlabel = f'{obj.time_name} {obj.time_unit}'
        else:
            xlabel = f'{obj.time_name}'
    else:
        xlabel = None
    
    if obj.value_name is not None:
        if obj.value_unit is not None:
            ylabel = f'{obj.value_name} {obj.value_unit}'
        else:
            ylabel = f'{obj.value_name}'
    else:
        ylabel=None

    return xlabel, ylabel

def bootstrap_fill_plot(series,ax=None,line_color=None,fill_color=None,transition_interval=None,
                        plot_kwargs = None, ci_kwargs = None,background_series=None,background_kwargs = None):
    '''Plot RQA results with bootstrap confidence bounds as a filled region

    Parameters
    ----------

    series : pyleoclim.Series
        Series object containing the timeseries to plot.

    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure will be generated.

    line_color : str or tuple, optional
        String or RGB tuple to use for the line colour.

    fill_color : str or tuple, optional
        String or RGB tuple to use for the fill colour between confidence bounds.

    transition_interval : list or tuple, optional
        (upper, lower) bounds for the transition interval. If None, computed
        automatically via bootstrapping using ci_kwargs.

    plot_kwargs : dict, optional
        Keyword arguments for the main plot.
        See `pyleoclim.Series.plot <https://pyleoclim-util.readthedocs.io/en/latest/core/api.html#series-pyleoclim-series>`_ for details.

    ci_kwargs : dict, optional
        Keyword arguments for calculating the confidence interval. Only used if
        transition_interval is not passed.
        See ammonyte.utils.sampling.confidence_interval for details.

    background_series : pyleoclim.Series, optional
        Optional series to plot behind the main series.

    background_kwargs : dict, optional
        Keyword arguments for the background plot. If not passed, the colour of the
        main plot will be reused and alpha will be set to 0.2.

    Returns
    -------

    fig : matplotlib.figure.Figure
        The figure object from matplotlib.
        See `matplotlib.pyplot.figure <https://matplotlib.org/stable/api/figure_api.html>`_ for details.

    ax : matplotlib.axes.Axes
        The axis object from matplotlib.
        See `matplotlib.axes <https://matplotlib.org/stable/api/axes_api.html>`_ for details.

    See Also
    --------

    ammonyte.utils.sampling.confidence_interval

    '''

    if not isinstance(series,pyleo.core.Series):
        raise ValueError('This function requires a pyleoclim.Series object, please reformat and pass again.')

    ci_kwargs = {} if ci_kwargs is None else ci_kwargs.copy()

    series = series.interp(step=1)

    if not transition_interval:
        transition_interval = confidence_interval(series,**ci_kwargs)
    
    if line_color is None:
        line_color=sns.color_palette('colorblind')[0]
    if fill_color is None:
        fill_color=sns.color_palette('colorblind')[0]
    
    #Need to find points in time where the series intersects with the lower and upper confidence boundaries
    value = series.value
    time = series.time
    
    idx_ufill = []
    idx_lfill = []

    upper = max(transition_interval)
    lower = min(transition_interval)

    #Need to find all points that lie above or below our confidence interval
    for idx,v in enumerate(value):
        if v >= upper:
            idx_ufill.append(idx)
        elif v <= lower:
            idx_lfill.append(idx)
    
    idx_sort = np.sort(np.concatenate((idx_ufill,idx_lfill)))
    
    if ax is None:
        fig,ax = plt.subplots(figsize=(12,8))

    if background_series:

        if not isinstance(background_series,pyleo.core.Series):
            raise ValueError('background_series does not appear to be a pyleoclim.Series object')
        
        #Standardize background_series values
        background_series.value -= np.mean(background_series.value)
        background_series.value /= max(background_series.value)

        #Adjust scaling to match that of main series
        background_series.value *= max(series.value - np.mean(series.value))
        background_series.value += np.mean(series.value)

        background_kwargs = {} if background_kwargs is None else background_kwargs.copy()

        if 'alpha' not in background_kwargs:
            background_kwargs['alpha'] = .2
        if 'color' not in background_kwargs:
            background_kwargs['color'] = line_color

        background_series.plot(ax=ax, **background_kwargs)

    series.plot(ax=ax,color=line_color)
        
    ufill_values = np.zeros(len(time)) + upper
    lfill_values = np.zeros(len(time)) + lower
    
    for i in range(len(time)):
        if i in idx_ufill:
            ufill_values[i] = value[i]
        elif i in idx_lfill:
            lfill_values[i] = value[i]
                                   
    ax.fill_between(time,lower,upper,color=fill_color,alpha=.1)
    ax.fill_between(time,upper,ufill_values,color=fill_color)
    ax.fill_between(time,lower,lfill_values,color=fill_color)
        
    if 'fig' in locals():
        return fig, ax
    else:
        return ax

def bootstrap_scatter_plot(series,ax=None,marker_color=None,transition_interval=None, scatter_kwargs= None,
                           ci_kwargs = None,background_series=None,background_kwargs = None):
    '''Plot RQA results highlighting points outside bootstrap confidence bounds as a scatter

    Parameters
    ----------

    series : pyleoclim.Series
        Series object containing the timeseries to plot.

    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure will be generated.

    marker_color : str or tuple, optional
        String or RGB tuple to use for the marker colour.

    transition_interval : list or tuple, optional
        (upper, lower) bounds for the transition interval. If None, computed
        automatically via bootstrapping using ci_kwargs.

    scatter_kwargs : dict, optional
        Keyword arguments for the scatter plot.

    ci_kwargs : dict, optional
        Keyword arguments for calculating the confidence interval. Only used if
        transition_interval is not passed.
        See ammonyte.utils.sampling.confidence_interval for details.

    background_series : pyleoclim.Series, optional
        Optional series to plot behind the main series.

    background_kwargs : dict, optional
        Keyword arguments for the background plot. If not passed, the colour of the
        main plot will be reused and alpha will be set to 0.2.

    Returns
    -------

    fig : matplotlib.figure.Figure
        The figure object from matplotlib.
        See `matplotlib.pyplot.figure <https://matplotlib.org/stable/api/figure_api.html>`_ for details.

    ax : matplotlib.axes.Axes
        The axis object from matplotlib.
        See `matplotlib.axes <https://matplotlib.org/stable/api/axes_api.html>`_ for details.

    See Also
    --------

    ammonyte.utils.sampling.confidence_interval

    '''

    if not isinstance(series,pyleo.core.Series):
        raise ValueError('This function requires a pyleoclim.Series object, please reformat and pass again.')

    ci_kwargs = {} if ci_kwargs is None else ci_kwargs.copy()
    scatter_kwargs = {} if scatter_kwargs is None else scatter_kwargs.copy()

    series = series.interp(step=1)

    if not transition_interval:
        transition_interval = confidence_interval(series,**ci_kwargs)
    
    if marker_color is None:
        line_color=sns.color_palette('colorblind')[0]
    
    value = series.value
    time = series.time
    
    idx_sig = []

    upper = max(transition_interval)
    lower = min(transition_interval)

    #Need to find all points that lie above or below our confidence interval
    for idx,v in enumerate(value):
        if v >= upper or v <= lower:
            idx_sig.append(idx)
    
    if ax is None:
        fig,ax = plt.subplots(figsize=(12,8))

    if background_series:

        if not isinstance(background_series,pyleo.core.Series):
            raise ValueError('background_series does not appear to be a pyleoclim.Series object')
        
        #Standardize background_series values
        background_series.value -= np.mean(background_series.value)
        background_series.value /= max(background_series.value)

        #Adjust scaling to match that of main series
        background_series.value *= max(series.value - np.mean(series.value))
        background_series.value += np.mean(series.value)

        background_kwargs = {} if background_kwargs is None else background_kwargs.copy()

        if 'alpha' not in background_kwargs:
            background_kwargs['alpha'] = .2
        if 'color' not in background_kwargs:
            background_kwargs['color'] = line_color

        background_series.plot(ax=ax, **background_kwargs)

    ax.scatter(time[idx_sig],value[idx_sig],c = marker_color, **scatter_kwargs)
        
    if 'fig' in locals():
        return fig, ax
    else:
        return ax