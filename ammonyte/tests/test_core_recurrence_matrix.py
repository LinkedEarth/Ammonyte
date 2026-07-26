''' Tests for ammonyte.core.recurrence_matrix
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
import ammonyte as amt
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class TestCoreRecurrenceMatrixLaplacianEigenmaps:
    '''Tests for laplacian eigenmaps function'''

    def test_laplacian_eigenmaps_t0(self, gen_series_with_transitions):
        '''laplacian_eigenmaps returns an RQARes carrying the Fisher Information series and propagated metadata'''
        ts_normal = gen_series_with_transitions(add_transitions=False, nt=100)
        td_sst = ts_normal.embed(3, 1)
        rm_sst = td_sst.create_recurrence_matrix(1)

        result = rm_sst.laplacian_eigenmaps(w_size=50, w_incre=5)

        # Check return type
        assert isinstance(result, amt.RQARes)

        # Check embedding/recurrence metadata is propagated
        assert result.m == 3
        assert result.tau == 1
        assert result.eps == 1
        assert result.w_size == 50
        assert result.w_incre == 5
        assert result.value_name == 'Fisher Information'
        assert result.series is ts_normal

        # Check Fisher Information series structure
        assert len(result.time) == len(result.value)
        assert len(result.time) > 0
        assert np.all(np.isfinite(result.value))

        # Check eigenmap is square and matches the recurrence matrix shape
        assert result.eigenmap.shape == rm_sst.matrix.shape


class TestCoreRecurrenceMatrixPlot:
    '''Tests for RecurrenceMatrix.plot'''

    def test_plot_returns_figure_axes_t0(self, gen_series_with_transitions):
        '''Test that plot runs without error and returns a figure and axes'''
        ts_normal = gen_series_with_transitions(add_transitions=False, nt=100)
        td_sst = ts_normal.embed(3, 1)
        rm_sst = td_sst.create_recurrence_matrix(1)

        fig, ax = rm_sst.plot()

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close('all')