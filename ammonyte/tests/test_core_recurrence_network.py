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


class TestCoreRecurrenceNetworkInit:
    '''Tests for RecurrenceNetwork initialization and attributes'''

    def test_create_recurrence_network_t0(self, gen_series_with_transitions):
        '''Test that create_recurrence_network returns a RecurrenceNetwork object'''
        ts = gen_series_with_transitions(add_transitions=False, nt=100)
        td = ts.embed(3, 1)

        rn = td.create_recurrence_network(1)

        assert isinstance(rn, RecurrenceNetwork)

    def test_recurrence_network_attributes_t0(self, gen_series_with_transitions):
        '''Test that RecurrenceNetwork stores the correct matrix, time, epsilon, and series'''
        ts = gen_series_with_transitions(add_transitions=False, nt=100)
        td = ts.embed(3, 1)

        rn = td.create_recurrence_network(1)

        assert rn.epsilon == 1
        assert rn.series is ts
        assert np.array_equal(rn.time, td.embedded_time)
        assert rn.matrix.shape == (len(rn.time), len(rn.time))

    def test_recurrence_network_no_embedding_params_t0(self, gen_series_with_transitions):
        '''Test that RecurrenceNetwork does not store m/tau, unlike RecurrenceMatrix

        This is a deliberate design choice (see RecurrenceNetwork docstring), and is the
        reason laplacian_eigenmaps must be blocked on RecurrenceNetwork (see
        TestCoreRecurrenceNetworkLaplacianEigenmaps below).'''
        ts = gen_series_with_transitions(add_transitions=False, nt=100)
        td = ts.embed(3, 1)

        rn = td.create_recurrence_network(1)

        assert not hasattr(rn, 'm')
        assert not hasattr(rn, 'tau')

    def test_recurrence_network_matrix_is_binary_t0(self, gen_series_with_transitions):
        '''Test that RecurrenceNetwork matrix contains only 0s and 1s'''
        ts = gen_series_with_transitions(add_transitions=False, nt=100)
        td = ts.embed(3, 1)

        rn = td.create_recurrence_network(1)

        assert np.all(np.isin(rn.matrix, [0, 1]))


class TestCoreRecurrenceNetworkLaplacianEigenmaps:
    '''Tests for laplacian_eigenmaps on RecurrenceNetwork'''

    def test_laplacian_eigenmaps_not_implemented_t0(self, gen_series_with_transitions):
        '''Test that laplacian_eigenmaps raises NotImplementedError on RecurrenceNetwork,
        since RecurrenceNetwork does not store the embedding parameters (m, tau) it needs'''
        ts = gen_series_with_transitions(add_transitions=False, nt=100)
        td = ts.embed(3, 1)

        rn = td.create_recurrence_network(1)

        with pytest.raises(NotImplementedError):
            rn.laplacian_eigenmaps(w_size=50, w_incre=5)