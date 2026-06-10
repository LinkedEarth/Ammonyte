''' Tests for ammonyte.core.recurrence_network
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
from ammonyte.core.recurrence_network import RecurrenceNetwork


def gen_normal(loc=0, scale=1, nt=100):
    '''Generate random data with a Gaussian distribution'''
    t = np.arange(nt)
    np.random.seed(42)
    v = np.random.normal(loc=loc, scale=scale, size=nt)
    return amt.Series(t, v)


class TestCoreRecurrenceNetworkInit:
    '''Tests for RecurrenceNetwork initialization and attributes'''

    def test_create_recurrence_network_t0(self):
        '''Test that create_recurrence_network returns a RecurrenceNetwork object'''
        ts = gen_normal()
        td = ts.embed(3, 1)

        rn = td.create_recurrence_network(1)

        assert isinstance(rn, RecurrenceNetwork)

    def test_recurrence_network_attributes_t0(self):
        '''Test that RecurrenceNetwork has expected attributes'''
        ts = gen_normal()
        td = ts.embed(3, 1)

        rn = td.create_recurrence_network(1)

        assert hasattr(rn, 'matrix')
        assert hasattr(rn, 'time')
        assert hasattr(rn, 'epsilon')

    def test_recurrence_network_matrix_is_binary_t0(self):
        '''Test that RecurrenceNetwork matrix contains only 0s and 1s'''
        ts = gen_normal()
        td = ts.embed(3, 1)

        rn = td.create_recurrence_network(1)

        assert np.all(np.isin(rn.matrix, [0, 1]))