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

from ..core.time_embedded_series import TimeEmbeddedSeries
from ..core.recurrence_matrix import RecurrenceMatrix
from ..utils.parameters import tau_search
from ..utils.ks import KS_test
from .transitions import DeterministicTransitions

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

        # Decode Unicode escapes in units (e.g., \u2030 → ‰)
        value_unit = pyleo_series.value_unit
        if value_unit and '\\u' in value_unit:
            value_unit = value_unit.encode().decode('unicode_escape')

        time_unit = pyleo_series.time_unit
        if time_unit and '\\u' in time_unit:
            time_unit = time_unit.encode().decode('unicode_escape')

        # Convert to ammonyte Series
        return cls(
            time=pyleo_series.time,
            value=pyleo_series.value,
            time_name=pyleo_series.time_name,
            value_name=pyleo_series.value_name,
            value_unit=value_unit,
            time_unit=time_unit,
            label=pyleo_series.label
        )

    def embed(self,m,tau=None,):
        '''Create a time delay embedding from an ammonyte.Series object

        Parameters
        ----------

        m : int
            Embedding dimension (number of delay coordinates).

        tau : int, optional
            Time delay for the embedding. If None, estimated automatically via the
            first minimum of mutual information using ammonyte.utils.parameters.tau_search.

        Returns
        -------

        TimeEmbeddedSeries : ammonyte.TimeEmbeddedSeries
            Time delay embedded representation of the series.

        See Also
        --------

        ammonyte.utils.parameters.tau_search

        ammonyte.TimeEmbeddedSeries
        '''

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
        from ..core.rqa_res import RQARes

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
        from ..core.rqa_res import RQARes

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

        ```python
        import os, ammonyte as amt
        ngrip = amt.Series.from_csv(os.path.join(os.path.dirname(amt.__file__), 'data', 'NGRIP.csv'))
        transitions = ngrip.kstest(w_min=0.12, w_max=2.5, n_w=15, d_c=0.77, n_c=3, s_c=2, x_c=0.8)
        print(transitions)
        ```

        Access transition statistics:

        ```python
        print(f"D-statistics: {transitions.d_statistics}")
        print(f"P-values: {transitions.p_values}")
        ```

        Plot the results:

        ```python
        transitions.plot()
        ```

        See Also
        --------
        DeterministicTransitions.plot : Visualize detected transitions
        
        References
        ----------
        
        .. [1] Bagniewski, W., Ghil, M., & Rousseau, D. D. (2021). 
               Automatic detection of abrupt transitions in paleoclimate records. 
               Chaos: An Interdisciplinary Journal of Nonlinear Science, 31(11), 113129.
               https://doi.org/10.1063/5.0062543
               
                      
        .. [2] Kolmogorov, A. (1933). Sulla determinazione empirica di una legge 
               di distribuzione. Giornale dell'Istituto Italiano degli Attuari, 4, 83-91.
               
        .. [3] Smirnov, N. (1948). Table for estimating the goodness of fit of 
               empirical distributions. The Annals of Mathematical Statistics, 19(2), 279-281.
        '''
        if not self.is_evenly_spaced():
            raise ValueError('This method requires evenly spaced data. '
                             'Use .interp(), .bin(), or .gkernel() prior to calling .kstest().')

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

    def ruptures(self, algo='Pelt', cost='rbf', pen=None, n_bkps=None,
                 min_size=2, jump=1, width=None, params=None):
        ''' Detect transitions using ruptures change point detection

        Applies ruptures algorithms for offline change point detection. 

        Parameters
        ----------
        algo : str
            Search algorithm. Default is 'Pelt'
            Options: 'Pelt', 'Dynp', 'Binseg', 'BottomUp', 'Window', 'KernelCPD'

        cost : str
            Cost function (type of change to detect). Default is 'rbf'
            Options: 'l1', 'l2', 'rbf', 'normal', 'ar', 'linear', 'rank', 'mahalanobis', 'cosine', 'clinear'

        pen : float, optional
            Penalty parameter controlling sensitivity to changepoints (higher = fewer changepoints)

            **What is penalty?**
            The algorithm minimizes: [fit error] + [number of changepoints × pen]
            - Low penalty → more changepoints (risk: overfitting, detecting noise as transitions)
            - High penalty → fewer changepoints (risk: missing real transitions)

            **Choosing penalty is a user decision** based on your data characteristics,
            expected transition frequency, and tolerance for false positives vs. false negatives.
            There is no single "correct" penalty value - it requires experimentation and
            domain knowledge.

            **Example approaches for selecting penalty:**
            1. **Fixed empirical values**: pen = 5, 10, 20, etc. (experiment to find what works)
            2. **Information criteria** (linear penalties that balance fit vs. complexity):
               - AIC (Akaike Information Criterion) - more liberal, detects more changepoints
               - BIC (Bayesian Information Criterion) - more conservative, sample-size dependent
               - mBIC (modified BIC) - even more conservative

               **Note**: The exact formula for each criterion depends on the statistical model
               and cost function used. For theoretical details, see ruptures documentation.

        n_bkps : int, optional
            Exact number of breakpoints to detect (alternative to penalty-based approach)
            Use when you know how many transitions to expect
            **Note**: Cannot be used together with 'pen' - choose one or the other

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
        DeterministicTransitions
            Object containing detected transitions with plot() and to_csv() methods

        Examples
        --------

        Basic transition detection:

        ```python
        import os, ammonyte as amt
        ngrip = amt.Series.from_csv(os.path.join(os.path.dirname(amt.__file__), 'data', 'NGRIP.csv'))
        transitions = ngrip.ruptures(algo='Pelt', cost='rbf', pen=5)
        print(transitions)
        ```

        Plot the results:

        ```python
        transitions.plot()
        ```


        References
        ----------

        .. [1] Truong, C., Oudre, L., & Vayatis, N. (2020).
               Selective review of offline change point detection methods.
               Signal Processing, 167, 107299.
        '''
        if not self.is_evenly_spaced():
            raise ValueError('This method requires evenly spaced data. '
                             'Use .interp(), .bin(), or .gkernel() prior to calling .ruptures().')

        from ..utils.ruptures_transitions import ruptures_transition

        return ruptures_transition(
            series=self,
            algo=algo,
            cost=cost,
            pen=pen,
            n_bkps=n_bkps,
            min_size=min_size,
            jump=jump,
            width=width,
            params=params
        )

