#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
import itertools

import pyleoclim as pyleo
import numpy as np

from tqdm import tqdm
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation

from ..core.rqa_res import RQARes
from ..core.time_embedded_series import TimeEmbeddedSeries
from ..core.recurrence_matrix import RecurrenceMatrix
from ..utils.parameters import tau_search
from ..utils.ks import KS_test
from ..utils.climate_phases import find_phases
from .transitions import DeterministicTransitions, ClimatePhases

class Series(pyleo.Series):
    '''Ammonyte series object, launching point for most ammonyte analysis.

    Child of pyleoclim.Series, so shares all methods with pyleoclim.Series plus those
    defined here.
    '''

    @classmethod
    def from_csv(cls, file_path):
        '''Load an ammonyte Series from a CSV file with pyleoclim metadata format.
        
        Parameters
        ----------
        file_path : str
            Path to the CSV file with pyleoclim metadata header format (### delimited)
            
        Returns
        -------
        ammonyte.Series
            An ammonyte Series object loaded from the CSV file
            
        Examples
        --------
        >>> ts = ammonyte.Series.from_csv('data/NGRIP.csv')
        '''
        # Use pyleoclim's from_csv to load the data
        pyleo_series = pyleo.Series.from_csv(file_path)
        
        # Convert to ammonyte Series
        return cls(
            time=pyleo_series.time, 
            value=pyleo_series.value,
            time_name=pyleo_series.time_name, 
            value_name=pyleo_series.value_name,
            value_unit=pyleo_series.value_unit, 
            time_unit=pyleo_series.time_unit,
            label=pyleo_series.label
        )

    def embed(self,m,tau=None,):
        '''Function to create a time delay embedding from a ammonyte.series object'''

        if tau is None:
            tau = tau_search(self)

        values = self.value
        time_axis = self.time[:(-m*tau)]
        
        manifold = np.ndarray(shape = (len(values)-(m*tau),m))

        for idx, _ in enumerate(values):
            if idx < (len(values)-(m*tau)):
                manifold[idx] = values[idx:idx+(m*tau):tau]

        embedded_data = manifold
        embedded_time = time_axis

        return TimeEmbeddedSeries(
            series=self,
            m=m,
            tau=tau,
            embedded_data=embedded_data,
            embedded_time=embedded_time,
            value_name=self.value_name,
            value_unit=self.value_unit,
            time_name=self.time_name,
            time_unit=self.time_unit,
            label=self.label)

    def determinism(self,window_size,overlap,m,tau,eps):
        '''Calculate determinism of a series

        Note that series must be evenly spaced for this method.
        See interp, bin, and gkernel methods in parent class pyleoclim.Series for details.
        
        Parameters
        ----------
        
        window_size : int
            Size of window to use when calculating recurrence plots for determinism statistic.
            Note this is in units of the time axis.
        
        overlap : int
            Amount of overlap to allow between windows.
            Note this is in units of the time axis.

        m : int
            Embedding dimension to use when performing time delay embedding,
            
        tau : int
            Time delay to use when performing time delay embedding
            
        eps : float
            Size of radius to use to calculate recurrence matrix

        Returns
        -------

        det_series : ammonyte.Series
            Ammonyte.Series object containing time series of the determinism statistic
        '''
       
        series = self
        windows = np.arange(int(min(series.time)),int(max(series.time)),int(overlap/2))

        cutoff_index = -int(window_size/(overlap/2))

        res = []
        window_time = []

        for window in tqdm(windows[:cutoff_index]):
            
            series_slice = series.slice((window,window+window_size))

            window_values = series_slice.value
            time = series_slice.time[int((len(series_slice.time)-1)/2)]

            ts = TimeSeries(window_values,
                            embedding_dimension = m,
                            time_delay=tau)

            settings = Settings(ts,
                                analysis_type=Classic,
                                neighbourhood=FixedRadius(eps),
                                similarity_measure=EuclideanMetric)

            computation = RQAComputation.create(settings,
                                                verbose=False)
            
            result = computation.run()

            window_time.append(time)

            res.append(result.determinism)

        det_series = RQARes(
            time = window_time,
            value = res,
            time_name=series.time_name,
            time_unit=series.time_unit,
            value_name='DET',
            label=series.label,
            m = m,
            tau = tau,
            eps = eps)

        return det_series

    def laminarity(self,window_size,overlap,m,tau,eps):
        '''Calculate laminarity of a series

        Note that series must be evenly spaced for this method.
        See interp, bin, and gkernel methods in parent class pyleoclim.Series for details.
        
        Parameters
        ----------
        
        window_size : int
            Size of window to use when calculating recurrence plots for determinism statistic.
            Note this is in units of the time axis.
        
        overlap : int
            Amount of overlap to allow between windows
            Note this is in units of the time axis.

        m : int
            Embedding dimension to use when performing time delay embedding,
            
        tau : int
            Time delay to use when performing time delay embedding
            
        eps : float
            Size of radius to use to calculate recurrence matrix

        Returns
        -------

        lam_series : ammonyte.Series
            Ammonyte.Series object containing time series of the laminarity statistic
        '''

        series = self
        windows = np.arange(int(min(series.time)),int(max(series.time)),int(overlap/2))

        cutoff_index = -int(window_size/(overlap/2))

        res = []
        window_time = []

        for window in tqdm(windows[:cutoff_index]):
            
            series_slice = series.slice((window,window+window_size))

            window_values = series_slice.value
            time = series_slice.time[int((len(series_slice.time)-1)/2)]

            ts = TimeSeries(window_values,
                            embedding_dimension = m,
                            time_delay=tau)

            settings = Settings(ts,
                                analysis_type=Classic,
                                neighbourhood=FixedRadius(eps),
                                similarity_measure=EuclideanMetric)

            computation = RQAComputation.create(settings,
                                                verbose=False)
            
            result = computation.run()

            window_time.append(time)

            res.append(result.laminarity)

        lam_series = RQARes(
            time=window_time,
            value=res,
            time_name=series.time_name,
            time_unit=series.time_unit,
            value_name='LAM',
            label=series.label,
            m = m,
            tau = tau,
            eps = eps)
        
        return lam_series

    def kstest(self, w_min, w_max, n_w=15, d_c=0.75, n_c=3, s_c=1.5, x_c=None):
        ''' Detect tipping points using Kolmogorov-Smirnov test
        
        Applies sliding window KS test to detect abrupt transitions in time series data using the method of Bagniewski et al. (2021).
        
        Parameters
        ----------
        w_min : float
            Size of smallest sliding window in time units
            
        w_max : float
            Size of largest sliding window in time units
            
        n_w : int
            Number of window lengths to test. Default is 15
            
        d_c : float
            Cut-off threshold for KS statistic. Default is 0.75
            
        n_c : int
            Minimum sample size per window. Default is 3
            
        s_c : float
            Standard deviation ratio threshold. Default is 1.5
            
        x_c : float
            Change threshold. If None, auto-calculated
        
        Returns
        -------
        DeterministicTransitions
            Object containing detected transitions with their corresponding 
            KS D-statistics and p-values, plus methods for analysis
        
        Examples
        --------
        
        Basic tipping point detection:
        
        .. jupyter-execute::
        
            import ammonyte as amt
            ngrip = amt.Series.from_csv('ammonyte/data/NGRIP.csv')
            transitions = ngrip.kstest(w_min=0.12, w_max=2.5, d_c=0.77)
            print(transitions)
        
        Access transition statistics:
        
        .. jupyter-execute::
        
            print(f"D-statistics: {transitions.d_statistics}")
            print(f"P-values: {transitions.p_values}")
        
        Plot the results:
        
        .. jupyter-execute::
        
            transitions.plot()
        
        See Also
        --------
        find_phases : Detect climate phase boundaries
        DeterministicTransitions.plot : Visualize detected transitions
        
        References
        ----------
        
        .. [1] Bagniewski, W., Ghil, M., & Rousseau, D. D. (2021). 
               Automatic detection of abrupt transitions in paleoclimate records. 
               Chaos: An Interdisciplinary Journal of Nonlinear Science, 31(11), 113129.
               https://doi.org/10.1063/5.0062543
        '''
        # Apply KS test algorithm
        jumps, d_statistics, p_values = KS_test(self, w_min, w_max, n_w, d_c, n_c, s_c, x_c)
        
        # Create result object with comprehensive metadata
        res = DeterministicTransitions(
            series=self, 
            jump_times=jumps[:,0], 
            jump_values=jumps[:,1], 
            method='KS test',
            method_args={
                'w_min': w_min, 'w_max': w_max, 'n_w': n_w, 
                'd_c': d_c, 'n_c': n_c, 's_c': s_c, 'x_c': x_c
            },
            label=getattr(self, 'label', None),
            statistics={'d_statistics': d_statistics, 'p_values': p_values}
        )
        return res

    def find_phases(self, window, interp_method=None):
        ''' Extract stadial and interstadial climate phase boundaries
        
        Identifies climate phases using sliding window approach to compute adaptive thresholds for Dansgaard-Oeschger event classification.
        
        Parameters
        ----------
        window : float
            Size of sliding window in time units
            
        interp_method : str, optional
            Interpolation method for final coordinate mapping ('linear', 'cubic', 'nearest', 'quadratic').
            **ONLY specify this if your original data was NOT evenly spaced and you interpolated it manually.**
            
            - If your data was originally evenly spaced: Leave as None (default)
            - If you used `series.interp()`: Use 'linear' 
            - If you used `series.interp()` with cubic: Use 'cubic'
            - If you used `series.gkernel()` or `series.bin()`: Use 'linear' or your preferred method
            
            This parameter should match the interpolation method you used during preprocessing.
            Default is None.
        
        Returns
        -------
        ClimatePhases
            Object containing climate phase boundaries and analysis methods
        
        Examples
        --------
        
        **Case 1: Originally evenly spaced data**
        
        .. jupyter-execute::
        
            import ammonyte as amt
            ngrip = amt.Series.from_csv('ammonyte/data/NGRIP.csv')  # Already evenly spaced
            phases = ngrip.find_phases(window=1.0)  # No interp_method needed
            print(phases)
        
        **Case 2: Originally irregular data (user interpolated)**
        
        .. jupyter-execute::
        
            import ammonyte as amt
            irregular_series = amt.Series.from_csv('irregular_data.csv')
            
            # User manually interpolates first
            even_series = irregular_series.interp(step=0.1)  # Linear interpolation
            
            # Must specify the interpolation method used
            phases = even_series.find_phases(window=1.0, interp_method='linear')
            print(phases)
        
        Plot climate phases:
        
        .. jupyter-execute::
        
            phases.plot()
        
        Extract phase periods:
        
        .. jupyter-execute::
        
            periods = phases.get_periods()
            print(f"Interstadial periods: {len(periods['interstadial'])}")
            print(f"Stadial periods: {len(periods['stadial'])}")
        
        Notes
        -----
        **Critical:** This method requires evenly spaced data. If your data is irregularly spaced, 
        you MUST interpolate it first using .interp(), .gkernel(), or .bin() methods.
        
        **Two scenarios handled:**
        
        1. **Originally evenly spaced data**: 
           - Call `find_phases(window=1.0)` with no interp_method
           - Algorithm uses computed values directly (no interpolation back)
           
        2. **Originally irregular data (user interpolated)**: 
           - First: manually interpolate your data 
           - Then: call `find_phases(window=1.0, interp_method='your_method')`
           - Algorithm will interpolate back using your specified method
        
        See Also
        --------
        kstest : Detect abrupt transitions
        
        References
        ----------
        Dansgaard, W., et al. (1993). Evidence for general instability of past climate from a 250-kyr ice-core record. Nature, 364(6434), 218-220.
        
        '''
        
        # Defensive approach: copy the original series to prevent accidental modification
        ts_copy = self.copy()
        
        # Check if data is evenly spaced before processing
        if not ts_copy.is_evenly_spaced():
            raise ValueError("Data is not evenly spaced. Please follow these steps:\n"
                           "1. Clean your data first: clean_data = data.clean()\n"
                           "2. Interpolate the cleaned data: interp_data = clean_data.interp()\n"
                           "3. Apply climate phases: interp_data.find_phases(window=X)")
        
        # Check if data is standardized (mean ≈ 0, std ≈ 1) 
        data_mean = np.mean(ts_copy.value)
        data_std = np.std(ts_copy.value, ddof=1)
        
        # More flexible tolerance for different data types:
        # - Ice core data (δ18O) typically has values around -40, std ~3
        # - Standardized data should have mean ≈ 0, std ≈ 1
        # Only flag as non-standardized if std is very different from 1 AND mean is not near expected range
        mean_tolerance = max(0.5, abs(data_mean) * 0.02)  # Allow 2% variation for large absolute means
        std_tolerance = 0.3
        
        is_likely_standardized = (abs(data_mean) < mean_tolerance and abs(data_std - 1.0) < std_tolerance)
        
        if is_likely_standardized:
            print("Data is evenly spaced and standardized - proceeding with climate phase detection.")
        else:
            raise ValueError(f"Data is not standardized (mean: {data_mean:.3f}, std: {data_std:.3f}). "
                           "Please standardize your data first:\n"
                           "1. Clean your data: clean_data = data.clean()\n"
                           "2. Interpolate: interp_data = clean_data.interp()\n"
                           "3. Standardize: std_data = interp_data.standardize()\n"
                           "4. Apply climate phases: std_data.find_phases(window=X)")
        
        # Determine if interpolation back is needed
        if interp_method is not None:
            print(f"Using {interp_method} interpolation for final coordinate mapping.")
            G_I, G_S = find_phases(ts_copy.time, ts_copy.value, window, interp_method, True)
        else:
            print("Using computed values directly (no interpolation back).")
            G_I, G_S = find_phases(ts_copy.time, ts_copy.value, window, None, False)
        
        res = ClimatePhases(
            series=self,
            interstadial_bounds=G_I,
            stadial_bounds=G_S, 
            method_args={'window': window, 'interp_method': interp_method},
            label=getattr(self, 'label', None)
        )
        return res

