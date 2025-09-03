''' Tests for ammonyte.utils.ks
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
from ammonyte.utils.ks import KS_test

import pyleoclim as pyleo


class TestUtilsKsBasic:
    '''Essential tests for KS_test function'''

    def test_ks_function_exists_t0(self):
        '''Test that KS_test function can be imported'''
        assert callable(KS_test)

    def test_ks_return_types_t0(self, gen_series_with_transitions):
        '''Test KS_test returns correct data types and structures'''
        ts = gen_series_with_transitions(add_transitions=True)
        
        jumps, d_statistics, p_values = KS_test(ts, w_min=0.5, w_max=2.0, n_w=15, 
                                               d_c=0.75, n_c=3, s_c=1.5, x_c=None)
        
        # Check return types
        assert isinstance(jumps, np.ndarray)
        assert isinstance(d_statistics, np.ndarray) 
        assert isinstance(p_values, np.ndarray)
        
        # Check array structures
        assert jumps.shape[1] == 2  # Should have time and direction columns
        assert len(d_statistics) == len(jumps)
        assert len(p_values) == len(jumps)

    def test_ks_value_ranges_t0(self, gen_series_with_transitions):
        '''Test returned values are in expected ranges'''
        ts = gen_series_with_transitions(add_transitions=True)
        jumps, d_stats, p_vals = KS_test(ts, 0.5, 2.0, 15, 0.75, 3, 1.5, None)
        
        if len(jumps) > 0 and not np.isnan(jumps[0, 0]):
            # Check direction values are ±1
            assert np.allclose(np.abs(jumps[:, 1]), 1.0)
            # Check D-statistics are in [0,1]
            assert np.all(d_stats >= 0) and np.all(d_stats <= 1)
            # Check p-values are in [0,1]
            assert np.all((p_vals >= 0) & (p_vals <= 1))



class TestUtilsKsIntegration:
    '''Essential integration tests'''

    def test_series_kstest_integration_t0(self, gen_series_with_transitions):
        '''Test integration between KS_test function and Series.kstest method'''
        ts = gen_series_with_transitions(add_transitions=True)
        
        transitions = ts.kstest(w_min=0.5, w_max=2.0, n_w=15, d_c=0.75, n_c=3, s_c=1.5, x_c=None)
        
        # Check result is DeterministicTransitions object
        from ammonyte.core.transitions import DeterministicTransitions
        assert isinstance(transitions, DeterministicTransitions)
        
        # Check new statistical attributes exist
        assert hasattr(transitions, 'd_statistics')
        assert hasattr(transitions, 'p_values')
        assert len(transitions.d_statistics) == len(transitions.jump_times)
        assert len(transitions.p_values) == len(transitions.jump_times)
        
        # Check method metadata
        assert transitions.method == 'KS test'
        assert 'w_min' in transitions.method_args

    def test_direct_vs_series_consistency_t0(self, gen_series_with_transitions):
        '''Test consistency between KS_test function and Series.kstest method'''
        ts = gen_series_with_transitions(add_transitions=True)
        params = dict(w_min=0.4, w_max=1.8, n_w=15, d_c=0.75, n_c=3, s_c=1.5, x_c=None)
        
        # Direct function call
        jumps_direct, d_stats_direct, p_vals_direct = KS_test(ts, **params)
        
        # Series method call
        transitions_series = ts.kstest(**params)
        
        # Results should be equivalent
        if len(jumps_direct) == 1 and np.isnan(jumps_direct[0, 0]):
            # No transitions case
            assert len(transitions_series.jump_times) == 1
            assert np.isnan(transitions_series.jump_times[0])
        else:
            # Compare results
            np.testing.assert_array_almost_equal(jumps_direct[:, 0], transitions_series.jump_times)
            np.testing.assert_array_almost_equal(jumps_direct[:, 1], transitions_series.jump_values)
            np.testing.assert_array_almost_equal(d_stats_direct, transitions_series.d_statistics)
            np.testing.assert_array_almost_equal(p_vals_direct, transitions_series.p_values)

