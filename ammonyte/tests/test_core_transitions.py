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
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

        result = make_transitions(ts, jump_times, jump_values, method='test_method',
                                   method_args={'w_min': 5}, statistics={'d_statistics': np.array([0.8, 0.6])})

        assert result.series is ts
        np.testing.assert_array_equal(result.jump_times, jump_times)
        np.testing.assert_array_equal(result.jump_values, jump_values)
        assert result.method == 'test_method'
        assert result.method_args == {'w_min': 5}
        np.testing.assert_array_equal(result.statistics['d_statistics'], [0.8, 0.6])

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

        np.testing.assert_array_equal(result.d_statistics, [0.8, 0.6])
        np.testing.assert_array_equal(result.p_values, [0.01, 0.05])


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

        # mutating the copy should not affect the original (deep copy)
        copied.jump_times[0] = 999.0
        assert result.jump_times[0] == 100.0


class TestCoreDeterministicTransitionsStr:
    '''Tests for DeterministicTransitions.__str__'''

    def test_str_with_transitions_t0(self, gen_series_with_transitions, capsys):
        '''Test __str__ prints method, transition counts, and per-transition details'''
        ts = gen_series_with_transitions()
        result = make_transitions(ts, [300.0, 700.0], [1, -1], method='ruptures')

        assert str(result) == ''
        printed = capsys.readouterr().out

        assert 'ruptures' in printed
        assert 'Transition Details:' in printed
        assert '300.00' in printed and 'Upward' in printed
        assert '700.00' in printed and 'Downward' in printed

    def test_str_no_transitions_t0(self, gen_series_with_transitions, capsys):
        '''Test __str__ reports zero transitions and skips the details section when none detected'''
        ts = gen_series_with_transitions()
        result = make_transitions(ts, [np.nan], [np.nan], method='ruptures')

        str(result)
        printed = capsys.readouterr().out

        assert 'ruptures' in printed
        # no per-transition detail lines should be printed when nothing was detected
        assert 'Transition Details:' not in printed
        assert 'Time:' not in printed


class TestCoreDeterministicTransitionsPlot:
    '''Tests for DeterministicTransitions.plot'''

    def test_plot_returns_figure_axes_t0(self, gen_series_with_transitions):
        '''Test that plot returns a figure and axes'''
        ts = gen_series_with_transitions()
        result = make_transitions(ts, [300.0, 700.0], [1, -1])

        fig, ax = result.plot()

        assert fig is not None
        assert ax is not None
        plt.close('all')

    def test_plot_no_transitions_t0(self, gen_series_with_transitions):
        '''Test that plot runs without error when no transitions detected'''
        ts = gen_series_with_transitions()
        result = make_transitions(ts, [np.nan], [np.nan])

        fig, ax = result.plot()
        assert fig is not None
        plt.close('all')

    @pytest.mark.parametrize('show_transitions', ['all', 'both', 'upward', 'downward'])
    def test_plot_show_transitions_options_t0(self, gen_series_with_transitions, show_transitions):
        '''Test that all show_transitions options run without error'''
        ts = gen_series_with_transitions()
        result = make_transitions(ts, [300.0, 700.0], [1, -1])

        result.plot(show_transitions=show_transitions)
        plt.close('all')

    def test_plot_show_legend_false_t0(self, gen_series_with_transitions):
        '''Test that plot runs without error when legend is disabled'''
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
        plt.close('all')


class TestCoreDeterministicTransitionsToCsv:
    '''Tests for DeterministicTransitions.to_csv'''

    def test_to_csv_writes_expected_data_t0(self, gen_series_with_transitions, tmp_path):
        '''Test that to_csv writes the expected columns and values, including statistics'''
        ts = gen_series_with_transitions()
        result = make_transitions(ts, [300.0, 700.0], [1, -1], method='test_method',
                                   statistics={'d_statistics': np.array([0.8, 0.6])})

        out_path = tmp_path / 'transitions.csv'
        returned_path = result.to_csv(path=str(out_path))

        assert returned_path == str(out_path)
        assert out_path.exists()

        df = pd.read_csv(out_path)
        np.testing.assert_array_equal(df['time'].values, [300.0, 700.0])
        np.testing.assert_array_equal(df['direction'].values, [1, -1])
        assert list(df['jump_type']) == ['upward_transition', 'downward_transition']
        assert list(df['method']) == ['test_method', 'test_method']
        np.testing.assert_array_equal(df['d_statistics'].values, [0.8, 0.6])

    def test_to_csv_no_transitions_t0(self, gen_series_with_transitions, tmp_path):
        '''Test that to_csv writes an empty table when no transitions were detected'''
        ts = gen_series_with_transitions()
        result = make_transitions(ts, [np.nan], [np.nan], method='test_method')

        out_path = tmp_path / 'transitions.csv'
        result.to_csv(path=str(out_path))

        df = pd.read_csv(out_path)
        assert len(df) == 0

    def test_to_csv_default_path_t0(self, gen_series_with_transitions, tmp_path, monkeypatch):
        '''Test that to_csv derives a default filename from the method name when path is None'''
        monkeypatch.chdir(tmp_path)
        ts = gen_series_with_transitions()
        result = make_transitions(ts, [300.0], [1], method='KS test')

        returned_path = result.to_csv()

        assert returned_path == 'KS_test_transitions.csv'
        assert (tmp_path / 'KS_test_transitions.csv').exists()
