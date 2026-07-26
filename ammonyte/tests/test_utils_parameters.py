''' Tests for ammonyte.utils.parameters
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
import pyleoclim as pyleo
import ammonyte as amt
import numpy as np

from ..utils.parameters import tau_search

class TestUtilsTauSearch:
    '''Tests for tau_search function, try a few different argument combinations
    '''

    @pytest.mark.parametrize('num_lags,return_MI',[(30,False),(30,True),(10,False)])
    def test_tau_search_t0(self,num_lags,return_MI, gen_series_with_transitions):
        '''Test tau search on gaussian series
        '''

        ts = gen_series_with_transitions(add_transitions=False, nt=100)

        if return_MI is False:
            tau = tau_search(ts,num_lags,return_MI)
        else:
            tau, MI = tau_search(ts,num_lags,return_MI)
            assert isinstance(MI, np.ndarray)
            assert len(MI) == num_lags - 1

        assert isinstance(tau, (int, np.integer))
        assert 1 <= tau <= num_lags - 1

    def test_tau_search_no_minimum_raises_t0(self):
        '''Test tau_search raises a clear ValueError when the mutual information
        curve has no local minimum and never reaches zero (e.g. a smooth,
        monotonically decaying series), instead of crashing with an
        UnboundLocalError.
        '''
        t = np.arange(300, dtype=float)
        v = np.exp(-t / 100.0)
        ts = amt.Series(time=t, value=v, time_unit='years', value_unit='x',
                        label='decay', auto_time_params=False)

        with pytest.raises(ValueError):
            tau_search(ts, num_lags=30)