#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import itertools

import pyleoclim as pyleo
import numpy as np
import pandas as pd
import multiprocessing as mp

from pyrqa.time_series import EmbeddedSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RPComputation

from ..core.recurrence_matrix import RecurrenceMatrix
from ..core.recurrence_network import RecurrenceNetwork
from ..utils.parameters import tau_search
from ..utils.range_finder import range_finder


class TimeEmbeddedSeries:
    '''Time embedded time series object. Precursor to recurrence matrix and recurrence network.

    Parameters
    ----------

    series : pyleoclim.Series or pandas.Series
        Time series to be embedded.

    m : int
        Embedding dimension.

    tau : int, optional
        Embedding delay. If None, calculated automatically via the first minimum
        of mutual information.

    embedded_data : numpy.ndarray, optional
        Pre-computed time delay embedded data. If not passed, will be calculated
        from series, m, and tau.

    embedded_time : array-like, optional
        Time axis corresponding to embedded_data. Must be passed alongside
        embedded_data if providing pre-computed data.

    value_name : str, optional
        Name of the value variable.

    value_unit : str, optional
        Units of the value variable.

    time_name : str, optional
        Name of the time variable.

    time_unit : str, optional
        Units of the time variable.

    label : str, optional
        Label for the object.
    '''

    def __init__(self,series,m,tau=None,embedded_data=None,embedded_time=None,value_name=None,value_unit=None,time_name=None,time_unit=None,label=None):
        self.series = series
        self.m = m
        self.tau = tau
        self.embedded_data = embedded_data
        self.embedded_time = embedded_time
        self.value_name = value_name
        self.value_unit = value_unit
        self.time_name = time_name
        self.time_unit = time_unit
        self.label = label

        if self.embedded_data is not None and self.embedded_time is None:
            raise ValueError('Embedded data was passed without associated time axis. Please pass neither or both')

        if self.tau is None:
            self.tau = tau_search(self.series)

        if self.embedded_data is None:

            if isinstance(self.series, (pyleo.Series, pyleo.GeoSeries)):
                values = self.series.value
                time_axis = self.series.time[:(-self.m*self.tau)]

            elif isinstance(self.series,pd.Series):
                values = self.series.values
                time_axis = list(self.series.index)[:(-self.m*self.tau)]

            else:
                raise ValueError('Unrecognized data type. Please pass a pyleoclim Series or pandas Series type object')
            
            manifold = np.ndarray(shape = (len(values)-(self.m*self.tau),self.m))

            for idx, i in enumerate(values):
                if idx < (len(values)-(self.m*self.tau)):
                    manifold[idx] = values[idx:idx+(self.m*self.tau):self.tau]

            self.embedded_data = manifold
            self.embedded_time = time_axis

        if self.value_name is None:
            self.value_name = self.series.value_name
        
        if self.value_unit is None:
            self.value_unit = self.series.value_unit

        if self.time_name is None:
            self.time_name = self.series.time_name
        
        if self.time_unit is None:
            self.time_unit = self.series.time_unit
        
        if self.label is None:
            self.label = self.series.label

    def create_recurrence_matrix(self,epsilon):
        '''Function to create Recurrence Matrix object
        
        Parameters
        ----------
        
        epsilon : float
            Fixed radius used to calculate whether two points are recurrent
            
        Returns
        -------
        
        RecurrenceMatrix : ammonyte.RecurrenceMatrix object'''

        ts = EmbeddedSeries(self.embedded_data)

        settings = Settings(ts,
                            analysis_type=Classic,
                            neighbourhood=FixedRadius(epsilon),
                            similarity_measure=EuclideanMetric)

        computation = RPComputation.create(settings,
                                        verbose=False)

        result = computation.run()

        matrix = result.recurrence_matrix

        return RecurrenceMatrix(
            matrix=matrix,
            time=self.embedded_time,
            epsilon=epsilon,
            series=self.series, 
            m = self.m, 
            tau = self.tau,
            value_name=self.value_name,
            value_unit=self.value_unit,
            time_name=self.time_name,
            time_unit=self.time_unit,
            label=self.label)

    def create_recurrence_network(self,epsilon):
        '''Function to create Recurrence Network object
        
        Parameters
        ----------
        
        epsilon : float
            Fixed radius used to calculate whether two points are recurrent.
            
        Returns
        -------
        
        RecurrenceNetwork : ammonyte.RecurrenceNetwork object'''

        ts = EmbeddedSeries(self.embedded_data)

        settings = Settings(ts,
                            analysis_type=Classic,
                            neighbourhood=FixedRadius(epsilon),
                            similarity_measure=EuclideanMetric)

        computation = RPComputation.create(settings,
                                        verbose=False)

        result = computation.run()

        matrix = result.recurrence_matrix

        return RecurrenceNetwork(
            matrix=matrix,
            time=self.embedded_time,
            epsilon=epsilon,
            series=self.series,
            value_name=self.value_name,
            value_unit=self.value_unit,
            time_name=self.time_name,
            time_unit=self.time_unit,
            label=self.label)

    def find_epsilon(self,eps,target_density=.05,tolerance=.01,initial_density=None,parallelize=False,num_processes=None,amp=10,verbose=True):
        '''Function to find epsilon value given target recurrence matrix density
        
        Parameters
        ----------

        eps : float
            Starting epsilon value (best guess).

        target_density : float
            Desired recurrence matrix density.

        tolerance : float
            Amount of allowable difference between target density and actual density.

        initial_density : float, optional
            If you have already calculated the initial density for your settings you can
            pass it here to save computation time.

        parallelize : bool; {True,False}
            Whether or not to parallelize the search process. Currently only tested on
            macOS, could be issues running this on Windows.

        num_processes : int, optional
            Number of processes to run. Automatically set to cpu count minus 2 if not passed.

        amp : int, optional
            Amplitude of the epsilon search range. Higher values cover ground quickly but
            converge more slowly. Default is 10.

        verbose : bool; {True,False}
            Whether or not to print output after each iteration. Default is True.

        Returns
        -------

        results : dict
            Dictionary with keys:

            - ``'Epsilon'`` : float — the epsilon value that produces the desired
              recurrence density within the specified tolerance.
            - ``'Output'`` : ammonyte.RecurrenceMatrix — the recurrence matrix computed
              at the converged epsilon value.

        See Also
        --------

        ammonyte.utils.range_finder.range_finder

        ammonyte.RecurrenceMatrix
        '''

        if num_processes is None:
            if mp.cpu_count() > 2:
                num_processes = mp.cpu_count() - 2
            else:
                num_processes = 1
        
        if initial_density is None:

            initial_result = self.create_recurrence_matrix(eps)
            initial_density = np.sum(initial_result.matrix)/np.size(initial_result.matrix)

            if verbose:
                print(f'Initial density is {initial_density:.4f}')
        
        if np.abs(initial_density - target_density) <= tolerance:

            if verbose:
                print('Initial density is within the tolerance window!')

            results = {'Epsilon':eps,'Output':initial_result}

            return results
        else:
            if verbose:
                print('Initial density is not within the tolerance window, searching...')
            density = initial_density

        if parallelize:

            while True:

                with mp.Pool(num_processes) as pool:
                    
                    eps_range, flag = range_finder(eps,density,target_density,tolerance,num_processes,amp)
                    
                    if flag is True:
                        
                        eps = eps_range
                        results = {'Epsilon':eps,'Output':self.create_recurrence_matrix(eps)}

                        if verbose:
                            matrix = results['Output'].matrix
                            density = np.sum(matrix)/np.size(matrix)
                            print(f'Epsilon: {eps:.4f}, Density: {density:.4f}.')

                        return results

                    r = pool.starmap(self.create_recurrence_matrix, zip(eps_range))
                    
                    pool.close()
                    pool.join()

                for item in r:
                    matrix = item.matrix
                    new_eps = item.epsilon
                    new_density = np.sum(matrix)/np.size(matrix)

                    if np.abs(new_density - .05) < np.abs(density -.05):
                        density = new_density
                        eps = new_eps

                if verbose:
        
                    print(f'Epsilon: {eps:.4f}, Density: {density:.4f}.')
        
        else:
            low_modifier=1
            high_modifier=1
            while True:

                distance = target_density-density

                if np.abs(distance) <= tolerance:
                        
                        results = {'Epsilon':eps,'Output':self.create_recurrence_matrix(eps)}

                        if verbose:
                            matrix = results['Output'].matrix
                            density = np.sum(matrix)/np.size(matrix)
                            print(f'Epsilon: {eps:.4f}, Density: {density:.4f}.')

                        return results

                new_eps = max(0,eps+(amp*distance*low_modifier*high_modifier))
                trial = self.create_recurrence_matrix(new_eps)
                matrix = trial.matrix
                new_eps = trial.epsilon
                new_density = np.sum(matrix)/np.size(matrix)
                new_distance = target_density - new_density

                if np.abs(new_distance) < np.abs(distance):
                    density = new_density
                    eps = new_eps
                    low_modifier=1
                    high_modifier=1

                elif (np.abs(new_distance) >= np.abs(distance)):
                    low_modifier /= 2

                if low_modifier < 1e-10:
                    raise RuntimeError('Runaway operation, exiting.')

                if verbose:
                    print(f'Epsilon: {eps:.4f}, Density: {density:.4f}')
