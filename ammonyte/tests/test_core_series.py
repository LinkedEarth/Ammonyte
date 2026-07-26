''' Tests for ammonyte.core.series
Naming rules:
1. class: Test{filename}{Class}{method} with appropriate camel case
2. function: test_{method}_t{test_id}

Notes on how to test:
0. Make sure [pytest](https://docs.pytest.org) has been installed: `pip install pytest`
1. execute `pytest {directory_path}` in terminal to perform all tests in all testing files inside the specified directory
    (certain tests will only work when run from the tests directory, so make sure to run from there!)
2. execute `pytest {file_path}` in terminal to perform all tests in the specified file
3. execute `pytest {file_path}::{TestClass}::{test_method}` in terminal to perform a specific test class/method inside the specified file
4. after `pip install pytest-xdist`, one may execute "pytest -n 4" to test in parallel with number of workers specified by `-n`
5. for more details, see https://docs.pytest.org/en/stable/usage.html
'''

import os
import pytest
import pyleoclim as pyleo
import ammonyte as amt
import numpy as np

class TestCoreSeriesEmbed:
    '''Tests for embed function
    '''

    @pytest.mark.parametrize('m,tau',[(10,5),(10,None)])
    def test_embed_t0(self,m,tau, gen_series_with_transitions):
        '''Test embed function with and without a tau value, on evenly spaced data'''

        ts = gen_series_with_transitions(add_transitions=False, nt=100)
        assert ts.is_evenly_spaced()

        result = ts.embed(m,tau)

        assert isinstance(result, amt.TimeEmbeddedSeries)
        assert result.series is ts
        assert result.m == m
        if tau is None:
            assert result.tau is not None
        else:
            assert result.tau == tau

    def test_embed_unevenly_spaced_raises_t0(self, gen_unevenly_spaced_series):
        '''Test embed raises ValueError for non-evenly spaced data'''
        ts = gen_unevenly_spaced_series()
        assert not ts.is_evenly_spaced()

        with pytest.raises(ValueError):
            ts.embed(3, 1)

class TestCoreSeriesDeterminism:
    '''Tests for determinism function
    '''

    @pytest.mark.parametrize('window_size,overlap,radius,m,tau',[(10,5,1,5,2),(12,4,.1,8,4)])
    def test_determinism_t0(self,window_size,overlap,m,tau,radius, gen_series_with_transitions):
        '''Test that determinism returns a well-formed RQARes'''
        ts = gen_series_with_transitions(add_transitions=False, nt=100)

        result = ts.determinism(window_size,overlap,m,tau,radius)

        assert isinstance(result, amt.RQARes)
        assert result.value_name == 'DET'
        assert result.m == m
        assert result.tau == tau
        assert result.eps == radius
        assert len(result.time) == len(result.value)
        assert len(result.time) > 0

    def test_determinism_unevenly_spaced_raises_t0(self, gen_unevenly_spaced_series):
        '''Test determinism raises ValueError for non-evenly spaced data'''
        ts = gen_unevenly_spaced_series()
        assert not ts.is_evenly_spaced()

        with pytest.raises(ValueError):
            ts.determinism(10,5,5,2,1)

class TestCoreSeriesLaminarity:
    '''Tests for laminarity function'''

    @pytest.mark.parametrize('window_size,overlap,radius,m,tau',[(10,5,1,5,2),(12,4,.1,8,4)])
    def test_laminarity_t0(self,window_size,overlap,m,tau,radius, gen_series_with_transitions):
        '''Test that laminarity returns a well-formed RQARes'''
        ts = gen_series_with_transitions(add_transitions=False, nt=100)

        result = ts.laminarity(window_size,overlap,m,tau,radius)

        assert isinstance(result, amt.RQARes)
        assert result.value_name == 'LAM'
        assert result.m == m
        assert result.tau == tau
        assert result.eps == radius
        assert len(result.time) == len(result.value)
        assert len(result.time) > 0

    def test_laminarity_unevenly_spaced_raises_t0(self, gen_unevenly_spaced_series):
        '''Test laminarity raises ValueError for non-evenly spaced data'''
        ts = gen_unevenly_spaced_series()
        assert not ts.is_evenly_spaced()

        with pytest.raises(ValueError):
            ts.laminarity(10,5,5,2,1)


class TestCoreSeriesFromCsv:
    '''Tests for Series.from_csv'''

    def test_from_csv_loads_series_t0(self):
        '''Test that from_csv loads a Series from the bundled NGRIP data file'''
        data_path = os.path.join(os.path.dirname(amt.__file__), 'data', 'NGRIP.csv')

        ts = amt.Series.from_csv(data_path)

        assert isinstance(ts, amt.Series)
        assert len(ts.time) > 0
        assert len(ts.value) > 0

    def test_from_csv_bad_path_t0(self):
        '''Test that from_csv raises an error for a non-existent file'''
        with pytest.raises(Exception):
            amt.Series.from_csv('nonexistent_file.csv')