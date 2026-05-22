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


class TestUtilsRupturesBasic:
    '''Essential tests for ruptures_transition function'''

    def test_ruptures_function_exists_t0(self):
        '''Test that ruptures_transition function can be imported'''
        assert callable(ruptures_transition)

    def test_ruptures_return_types_t0(self, gen_series_with_transitions):
        '''Test ruptures_transition returns correct data types and structures'''
        ts = gen_series_with_transitions(add_transitions=True)

        result = ruptures_transition(ts, algo='Pelt', model='rbf', pen=5)

        # Check return type
        from ammonyte.core.transitions import DeterministicTransitions
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
        result = ruptures_transition(ts, algo='Pelt', model='rbf', pen=5)

        if len(result.jump_times) > 0 and not np.isnan(result.jump_times[0]):
            # Check direction values are -1, 0, or +1
            assert np.all(np.isin(result.jump_values, [-1, 0, 1]))
            # Check method is set correctly
            assert result.method == 'ruptures'


class TestUtilsRupturesIntegration:
    '''Essential integration tests'''

    def test_series_ruptures_integration_t0(self, gen_series_with_transitions):
        '''Test integration between ruptures_transition function and Series.ruptures method'''
        ts = gen_series_with_transitions(add_transitions=True)

        transitions = ts.ruptures(algo='Pelt', model='rbf', pen=5)

        # Check result is DeterministicTransitions object
        from ammonyte.core.transitions import DeterministicTransitions
        assert isinstance(transitions, DeterministicTransitions)

        # Check method metadata
        assert transitions.method == 'ruptures'
        assert 'algo' in transitions.method_args
        assert 'model' in transitions.method_args
        assert 'pen' in transitions.method_args

        # Check statistics exist
        assert 'breakpoint_indices' in transitions.statistics

    def test_direct_vs_series_consistency_t0(self, gen_series_with_transitions):
        '''Test consistency between ruptures_transition function and Series.ruptures method'''
        ts = gen_series_with_transitions(add_transitions=True)
        params = dict(algo='Pelt', model='rbf', pen=5)

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
