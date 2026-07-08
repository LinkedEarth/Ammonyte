''' Tests for ammonyte.utils.lerm_transitions
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
from ammonyte.utils.lerm_transitions import lerm_transition


class TestUtilsLermBasic:
    '''Essential tests for lerm_transition function'''

    def test_lerm_function_exists_t0(self):
        '''Test that lerm_transition function can be imported'''
        assert callable(lerm_transition)

    def test_lerm_return_types_t0(self, gen_smooth_series):
        '''Test lerm_transition returns correct types'''
        ts = gen_smooth_series(nt=300)

        jump_times, jump_directions, upper_bound, lower_bound = lerm_transition(
            ts, upper=95, lower=5, w=50, n_samples=500
        )

        assert isinstance(jump_times, np.ndarray)
        assert isinstance(jump_directions, np.ndarray)
        assert isinstance(upper_bound, float)
        assert isinstance(lower_bound, float)

    def test_lerm_bounds_ordering_t0(self, gen_smooth_series):
        '''Test that upper bound is greater than lower bound'''
        ts = gen_smooth_series(nt=300)

        _, _, upper_bound, lower_bound = lerm_transition(
            ts, upper=95, lower=5, w=50, n_samples=500
        )

        assert upper_bound > lower_bound

    def test_lerm_directions_valid_t0(self, gen_smooth_series):
        '''Test that detected directions are only +1 or -1'''
        ts = gen_smooth_series(nt=300)

        _, jump_directions, _, _ = lerm_transition(
            ts, upper=95, lower=5, w=50, n_samples=500
        )

        if len(jump_directions) > 0:
            assert np.all(np.isin(jump_directions, [-1, 1]))

    def test_lerm_with_transition_interval_t0(self, gen_smooth_series):
        '''Test lerm_transition with explicitly provided transition interval'''
        ts = gen_smooth_series(nt=300)

        jump_times, jump_directions, upper_bound, lower_bound = lerm_transition(
            ts, transition_interval=(0.8, 0.2)
        )

        assert upper_bound == 0.8
        assert lower_bound == 0.2
