''' Shared test fixtures for ammonyte testing

This module contains shared pytest fixtures for synthetic data generation.
Following best practices for scientific Python packages like Pyleoclim.
'''

import pytest
import numpy as np
import ammonyte as amt
import pyleoclim as pyleo


@pytest.fixture
def random_seed():
    '''Provide the canonical random seed for reproducible synthetic test data'''
    return 251


@pytest.fixture
def gen_series_with_transitions(random_seed):
    '''Generate time series with artificial transitions at 30% and 70% positions'''
    def _gen(add_transitions=True, nt=800, seed=random_seed):
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
def gen_smooth_series(random_seed):
    '''Generate smooth sinusoidal series, standing in for a smoothed Fisher Information series'''
    def _gen(add_transitions=False, nt=500, seed=random_seed):
        '''Generate smooth series with optional step transitions at 30% and 70% positions'''
        np.random.seed(seed)
        t = np.linspace(0, 10, nt)
        v = np.sin(0.5 * t) + np.random.normal(0, 0.1, nt)

        if add_transitions:
            # Add step transitions at 30% and 70% through series
            idx1, idx2 = int(0.3 * len(v)), int(0.7 * len(v))
            v[idx1:] += 2.0
            v[idx2:] -= 1.5

        label = 'Smooth Data with transitions' if add_transitions else 'Smooth Data'
        return amt.Series(time=t, value=v, time_unit='years', value_unit='proxy_units',
                         label=label, auto_time_params=False)
    return _gen


@pytest.fixture
def gen_flat_series():
    '''Generate a constant, zero-variance Series with no detectable structure'''
    def _gen(nt=200, value=0.0):
        '''Generate a flat series, for testing that methods correctly detect nothing'''
        t = np.arange(nt, dtype=float)
        v = np.full(nt, value, dtype=float)
        return amt.Series(time=t, value=v, time_unit='years', value_unit='proxy_units',
                         label='Flat Data', auto_time_params=False)
    return _gen


@pytest.fixture
def gen_unevenly_spaced_series(random_seed):
    '''Generate a Series with irregularly spaced time points'''
    def _gen(nt=100, seed=random_seed):
        '''Generate an unevenly spaced series, for testing methods that require evenly spaced data'''
        np.random.seed(seed)
        t = pyleo.utils.random_time_axis(nt, delta_t_dist='exponential', param=[1.0], seed=seed)
        v = np.random.normal(size=nt)
        return amt.Series(time=t, value=v, time_unit='years', value_unit='proxy_units',
                         label='Unevenly Spaced Data', auto_time_params=False)
    return _gen


@pytest.fixture
def gen_geoseries_with_transitions(random_seed):
    '''Generate a GeoSeries with artificial transitions at 30% and 70% positions'''
    def _gen(add_transitions=True, nt=800, lat=75.1, lon=-42.32, seed=random_seed):
        '''Generate a GeoSeries with optional transitions on white noise background'''
        np.random.seed(seed)

        # Use white noise for clean algorithm testing
        t, v = pyleo.utils.gen_ts(model="colored_noise", alpha=0.0, nt=nt, seed=seed)

        if add_transitions:
            # Add step transitions at 30% and 70% through series
            idx1, idx2 = int(0.3 * len(v)), int(0.7 * len(v))
            v[idx1:] += 2.0
            v[idx2:] -= 1.5

        label = 'Test Data with transitions' if add_transitions else 'Test Data'
        return amt.GeoSeries(time=t, value=v, lat=lat, lon=lon, time_unit='years',
                            value_unit='proxy_units', label=label, auto_time_params=False)
    return _gen