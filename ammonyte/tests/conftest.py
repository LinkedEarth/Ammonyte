''' Shared test fixtures for ammonyte testing

This module contains shared pytest fixtures for synthetic data generation.
Following best practices for scientific Python packages like Pyleoclim.
'''

import pytest
import numpy as np
import ammonyte as amt
import pyleoclim as pyleo


@pytest.fixture
def gen_series_with_transitions():
    '''Generate time series with artificial transitions at 30% and 70% positions'''
    def _gen(add_transitions=True, nt=800, seed=251):
        '''Generate series with optional transitions on white noise background'''
        np.random.seed(seed)
        
        # Use white noise for clean algorithm testing
        t, v = pyleo.utils.gen_ts(model="colored_noise", alpha=0.0, nt=nt, seed=seed)
            
        if add_transitions:
            # Add step transitions at 30% and 70% through series
            idx1, idx2 = int(0.3 * len(v)), int(0.7 * len(v))
            v[idx1:] += 2.0  
            v[idx2:] -= 1.5  
        
        label = 'Test Data with transitions' if add_transitions else 'Test Data'
        return amt.Series(time=t, value=v, time_unit='years', value_unit='proxy_units',
                         label=label, auto_time_params=False)
    return _gen


@pytest.fixture
def gen_smooth_series():
    '''Generate smooth sinusoidal series without transitions'''
    def _gen(nt=500, seed=252):
        '''Generate smooth series without transitions'''
        np.random.seed(seed)
        t = np.linspace(0, 10, nt)
        v = np.sin(0.5 * t) + np.random.normal(0, 0.1, nt)
        return amt.Series(time=t, value=v, time_unit='years', value_unit='proxy_units',
                         label='Smooth Data', auto_time_params=False)
    return _gen