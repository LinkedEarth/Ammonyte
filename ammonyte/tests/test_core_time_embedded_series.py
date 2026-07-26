''' Tests for ammonyte.core.time_embedded_series
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

import pytest
import numpy as np
import pandas as pd
import ammonyte as amt


class TestCoreTimeEmbeddSeriesInit:
    '''Tests for TimeEmbeddedSeries instantiation'''

    def test_embedding_computation_t0(self):
        '''Test that embedded_data is computed correctly for a known series'''
        t = np.arange(10, dtype=float)
        v = np.arange(10, dtype=float) * 10
        ts = amt.Series(time=t, value=v, time_unit='years', value_unit='u',
                         label='x', auto_time_params=False)

        td = ts.embed(3, 1)

        assert td.embedded_data.shape == (7, 3)
        np.testing.assert_array_equal(td.embedded_data[0], [0, 10, 20])
        np.testing.assert_array_equal(td.embedded_data[-1], [60, 70, 80])
        np.testing.assert_array_equal(td.embedded_time, np.arange(7))

    def test_metadata_defaults_from_series_t0(self):
        '''Test that value_unit/time_unit/label default from the source series when not given'''
        ts = amt.Series(time=np.arange(10, dtype=float), value=np.arange(10, dtype=float),
                         time_unit='years', value_unit='proxy_units', label='mylabel',
                         auto_time_params=False)

        td = ts.embed(3, 1)

        assert td.value_unit == ts.value_unit
        assert td.time_unit == ts.time_unit
        assert td.label == ts.label

    def test_tau_none_auto_search_t0(self, gen_series_with_transitions):
        '''Test that tau is automatically estimated when not provided'''
        ts = gen_series_with_transitions(add_transitions=False, nt=100)

        td = ts.embed(10, None)

        assert td.tau is not None
        assert td.tau > 0

    def test_embedded_data_without_time_raises_t0(self):
        '''Test ValueError is raised when embedded_data is passed without embedded_time'''
        with pytest.raises(ValueError):
            amt.TimeEmbeddedSeries(series=None, m=3, tau=1, embedded_data=np.zeros((5, 3)))

    def test_unrecognized_series_type_raises_t0(self):
        '''Test ValueError is raised for a series type that is neither pyleoclim nor pandas'''
        with pytest.raises(ValueError):
            amt.TimeEmbeddedSeries(series='not a series', m=3, tau=1)

    def test_pandas_series_input_t0(self):
        '''Test that a plain pandas.Series can be embedded, with metadata left as given'''
        t = np.arange(10, dtype=float)
        v = np.arange(10, dtype=float) * 10
        pds = pd.Series(data=v, index=t)

        td = amt.TimeEmbeddedSeries(pds, m=3, tau=1)

        assert td.embedded_data.shape == (7, 3)
        np.testing.assert_array_equal(td.embedded_data[0], [0, 10, 20])
        assert td.value_name is None
        assert td.label is None


class TestCoreTimeEmbeddSeriesCreateRecurrenceMatrix:
    '''Tests for create_recurrence_matrix
    '''

    def test_create_recurrence_matrix_t0(self, gen_series_with_transitions):
        '''Test that create_recurrence_matrix returns a well-formed RecurrenceMatrix'''
        ts_normal = gen_series_with_transitions(add_transitions=False, nt=100)
        td_sst = ts_normal.embed(3, 3)

        result = td_sst.create_recurrence_matrix(1)

        assert isinstance(result, amt.RecurrenceMatrix)
        assert result.m == 3
        assert result.tau == 3
        assert result.epsilon == 1
        assert result.series is ts_normal
        assert np.array_equal(result.time, td_sst.embedded_time)
        assert result.matrix.shape == (len(result.time), len(result.time))
        assert np.all(np.isin(result.matrix, [0, 1]))


class TestCoreTimeEmbeddSeriesCreateRecurrenceNetwork:
    '''Tests for create_recurrence_network
    '''

    def test_create_recurrence_network_t0(self, gen_series_with_transitions):
        '''Test that create_recurrence_network returns a well-formed RecurrenceNetwork'''
        ts_normal = gen_series_with_transitions(add_transitions=False, nt=100)
        td_sst = ts_normal.embed(3, 3)

        result = td_sst.create_recurrence_network(1)

        assert isinstance(result, amt.RecurrenceNetwork)
        assert result.epsilon == 1
        assert result.series is ts_normal
        assert np.array_equal(result.time, td_sst.embedded_time)
        assert result.matrix.shape == (len(result.time), len(result.time))
        assert np.all(np.isin(result.matrix, [0, 1]))


class TestCoreTimeEmbeddSeriesFindEpsilon:
    '''Tests for find_epsilon

    Only the parallelize=False path is tested. parallelize=True spins up a
    multiprocessing.Pool, which is slow (~5s+ even on a tiny series) and, per
    its own docstring, "currently only tested on macOS" - not a good fit for
    the regular test suite.
    '''

    @pytest.mark.parametrize('eps', [.1, 1])
    def test_find_eps_t0(self, eps, gen_series_with_transitions):
        '''Test that find_epsilon converges to the target recurrence matrix density'''
        ts_normal = gen_series_with_transitions(add_transitions=False, nt=100)
        td = ts_normal.embed(3, 1)

        target_density = 0.05
        tolerance = 0.01
        result = td.find_epsilon(eps, target_density=target_density, tolerance=tolerance,
                                  parallelize=False, verbose=False)

        assert isinstance(result, dict)
        assert set(result.keys()) == {'Epsilon', 'Output'}
        assert isinstance(result['Output'], amt.RecurrenceMatrix)
        assert result['Output'].epsilon == result['Epsilon']

        density = np.sum(result['Output'].matrix) / np.size(result['Output'].matrix)
        assert np.abs(density - target_density) <= tolerance
