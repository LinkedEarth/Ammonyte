''' Tests for ammonyte.utils.ruptures_transitions
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
import ammonyte as amt
from ammonyte.utils.ruptures_transitions import ruptures_transition
from ammonyte.core.transitions import DeterministicTransitions


class TestUtilsRupturesBasic:
    '''Essential tests for ruptures_transition function'''

    def test_ruptures_function_exists_t0(self):
        '''Test that ruptures_transition function can be imported'''
        assert callable(ruptures_transition)

    def test_ruptures_return_types_t0(self, gen_series_with_transitions):
        '''Test ruptures_transition returns correct data types and structures'''
        ts = gen_series_with_transitions(add_transitions=True)

        result = ruptures_transition(ts, algo='Pelt', cost='rbf', pen=5)

        # Check return type
        assert isinstance(result, DeterministicTransitions)

        # Check attributes exist
        assert hasattr(result, 'jump_times')
        assert hasattr(result, 'jump_values')
        assert hasattr(result, 'method')
        assert hasattr(result, 'method_args')
        assert hasattr(result, 'statistics')

    def test_ruptures_value_ranges_t0(self, gen_series_with_transitions):
        '''Test returned values are in expected ranges'''
        ts = gen_series_with_transitions(add_transitions=True)
        result = ruptures_transition(ts, algo='Pelt', cost='rbf', pen=5)

        if len(result.jump_times) > 0 and not np.isnan(result.jump_times[0]):
            # Check direction values are -1, 0, or +1
            assert np.all(np.isin(result.jump_values, [-1, 0, 1]))
            # Check method is set correctly
            assert result.method == 'ruptures'


class TestUtilsRupturesValidation:
    '''Tests for invalid parameter combinations raising ValueError'''

    @pytest.mark.parametrize('kwargs,match', [
        (dict(algo='Pelt', cost='rbf', pen=5, n_bkps=2), "Cannot provide both"),
        (dict(algo='Pelt', cost='rbf'), "requires 'pen'"),
        (dict(algo='Dynp', cost='l2'), "requires 'n_bkps'"),
        (dict(algo='Window', cost='l2', n_bkps=2), "requires 'width'"),
        (dict(algo='KernelCPD', cost='l1', n_bkps=2), "only supports"),
        (dict(algo='NotARealAlgo', cost='rbf', pen=5), "Unknown algorithm"),
    ])
    def test_ruptures_invalid_params_raise_t0(self, gen_series_with_transitions, kwargs, match):
        '''Test that invalid parameter combinations raise ValueError with a helpful message'''
        ts = gen_series_with_transitions(add_transitions=True)
        with pytest.raises(ValueError, match=match):
            ruptures_transition(ts, **kwargs)


class TestUtilsRupturesIntegration:
    '''Essential integration tests'''

    def test_series_ruptures_integration_t0(self, gen_series_with_transitions):
        '''Test integration between ruptures_transition function and Series.ruptures method'''
        ts = gen_series_with_transitions(add_transitions=True)
        assert ts.is_evenly_spaced()

        transitions = ts.ruptures(algo='Pelt', cost='rbf', pen=5)

        # Check result is DeterministicTransitions object
        assert isinstance(transitions, DeterministicTransitions)

        # Check method metadata
        assert transitions.method == 'ruptures'
        assert 'algo' in transitions.method_args
        assert 'cost' in transitions.method_args
        assert 'pen' in transitions.method_args

        # Check statistics exist
        assert 'breakpoint_indices' in transitions.statistics

    def test_ruptures_unevenly_spaced_raises_t0(self, gen_unevenly_spaced_series):
        '''Test Series.ruptures raises ValueError for non-evenly spaced data'''
        ts = gen_unevenly_spaced_series()
        assert not ts.is_evenly_spaced()

        with pytest.raises(ValueError):
            ts.ruptures(algo='Pelt', cost='rbf', pen=5)

    def test_ruptures_flat_finds_nothing_t0(self, gen_flat_series):
        '''Test Series.ruptures detects no transitions on a flat, zero-variance series'''
        ts = gen_flat_series()

        result = ts.ruptures(algo='Pelt', cost='rbf', pen=5)

        assert len(result.jump_times) == 1
        assert np.isnan(result.jump_times[0])

    def test_ruptures_detects_injected_transitions_t0(self, gen_series_with_transitions):
        '''Test Series.ruptures finds the injected transitions on a series with known transitions'''
        ts = gen_series_with_transitions(add_transitions=True)

        result = ts.ruptures(algo='Pelt', cost='rbf', pen=10)

        assert len(result.jump_times) > 0
        assert np.all(np.isin(result.jump_values, [-1, 0, 1]))

    def test_direct_vs_series_consistency_t0(self, gen_series_with_transitions):
        '''Test consistency between ruptures_transition function and Series.ruptures method'''
        ts = gen_series_with_transitions(add_transitions=True)
        params = dict(algo='Pelt', cost='rbf', pen=5)

        # Direct function call
        result_direct = ruptures_transition(ts, **params)

        # Series method call
        result_series = ts.ruptures(**params)

        # Results should be equivalent
        if len(result_direct.jump_times) == 1 and np.isnan(result_direct.jump_times[0]):
            # No transitions case
            assert len(result_series.jump_times) == 1
            assert np.isnan(result_series.jump_times[0])
        else:
            # Compare results
            np.testing.assert_array_almost_equal(result_direct.jump_times, result_series.jump_times)
            np.testing.assert_array_almost_equal(result_direct.jump_values, result_series.jump_values)
