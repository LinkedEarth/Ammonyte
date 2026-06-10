''' Tests for ammonyte.core.transitions
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
import matplotlib
matplotlib.use('Agg')
import ammonyte as amt
from ammonyte.core.transitions import DeterministicTransitions


def make_transitions(series, jump_times, jump_values, method='test_method',
                     method_args=None, statistics=None):
    '''Helper to create a DeterministicTransitions object'''
    return DeterministicTransitions(
        series=series,
        jump_times=jump_times,
        jump_values=jump_values,
        method=method,
        method_args=method_args or {},
        statistics=statistics or {}
    )


class TestCoreDeterministicTransitionsInit:
    '''Tests for DeterministicTransitions initialization'''

    def test_init_attributes_t0(self, gen_series_with_transitions):
        '''Test that all attributes are set correctly on initialization'''
        ts = gen_series_with_transitions(add_transitions=True)
        jump_times = np.array([300.0, 700.0])
        jump_values = np.array([1, -1])

        result = make_transitions(ts, jump_times, jump_values)

        assert hasattr(result, 'series')
        assert hasattr(result, 'jump_times')
        assert hasattr(result, 'jump_values')
        assert hasattr(result, 'method')
        assert hasattr(result, 'method_args')
        assert hasattr(result, 'statistics')

    def test_init_array_conversion_t0(self, gen_series_with_transitions):
        '''Test that jump_times and jump_values are converted to numpy arrays'''
        ts = gen_series_with_transitions()
        result = make_transitions(ts, [100.0, 200.0], [1, -1])

        assert isinstance(result.jump_times, np.ndarray)
        assert isinstance(result.jump_values, np.ndarray)

    def test_init_statistics_attributes_t0(self, gen_series_with_transitions):
        '''Test that statistics dict entries become direct attributes'''
        ts = gen_series_with_transitions()
        stats = {'d_statistics': np.array([0.8, 0.6]), 'p_values': np.array([0.01, 0.05])}
        result = make_transitions(ts, [100.0, 200.0], [1, -1], statistics=stats)

        assert hasattr(result, 'd_statistics')
        assert hasattr(result, 'p_values')


class TestCoreDeterministicTransitionsCopy:
    '''Tests for DeterministicTransitions.copy'''

    def test_copy_t0(self, gen_series_with_transitions):
        '''Test that copy returns an independent object'''
        ts = gen_series_with_transitions()
        result = make_transitions(ts, [100.0], [1])
        copied = result.copy()

        assert copied is not result
        np.testing.assert_array_equal(copied.jump_times, result.jump_times)
        np.testing.assert_array_equal(copied.jump_values, result.jump_values)


class TestCoreDeterministicTransitionsStr:
    '''Tests for DeterministicTransitions.__str__'''

    def test_str_with_transitions_t0(self, gen_series_with_transitions):
        '''Test __str__ runs without error when transitions exist'''
        ts = gen_series_with_transitions()
        result = make_transitions(ts, [300.0, 700.0], [1, -1], method='ruptures')
        str(result)

    def test_str_no_transitions_t0(self, gen_series_with_transitions):
        '''Test __str__ runs without error when no transitions detected'''
        ts = gen_series_with_transitions()
        result = make_transitions(ts, [np.nan], [np.nan])
        str(result)


class TestCoreDeterministicTransitionsPlot:
    '''Tests for DeterministicTransitions.plot'''

    def test_plot_returns_figure_axes_t0(self, gen_series_with_transitions):
        '''Test that plot returns a figure and axes'''
        import matplotlib.pyplot as plt
        ts = gen_series_with_transitions()
        result = make_transitions(ts, [300.0, 700.0], [1, -1])

        fig, ax = result.plot()

        assert fig is not None
        assert ax is not None
        plt.close('all')

    def test_plot_no_transitions_t0(self, gen_series_with_transitions):
        '''Test that plot runs without error when no transitions detected'''
        import matplotlib.pyplot as plt
        ts = gen_series_with_transitions()
        result = make_transitions(ts, [np.nan], [np.nan])

        fig, ax = result.plot()
        assert fig is not None
        plt.close('all')

    @pytest.mark.parametrize('show_transitions', ['all', 'both', 'upward', 'downward'])
    def test_plot_show_transitions_options_t0(self, gen_series_with_transitions, show_transitions):
        '''Test that all show_transitions options run without error'''
        import matplotlib.pyplot as plt
        ts = gen_series_with_transitions()
        result = make_transitions(ts, [300.0, 700.0], [1, -1])

        result.plot(show_transitions=show_transitions)
        plt.close('all')

    def test_plot_show_legend_false_t0(self, gen_series_with_transitions):
        '''Test that plot runs without error when legend is disabled'''
        import matplotlib.pyplot as plt
        ts = gen_series_with_transitions()
        result = make_transitions(ts, [300.0], [1])

        result.plot(show_legend=False)
        plt.close('all')

    def test_plot_invalid_show_transitions_t0(self, gen_series_with_transitions):
        '''Test that plot raises ValueError for invalid show_transitions value'''
        ts = gen_series_with_transitions()
        result = make_transitions(ts, [300.0], [1])

        with pytest.raises(ValueError):
            result.plot(show_transitions='invalid')
