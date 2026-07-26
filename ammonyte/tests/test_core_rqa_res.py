''' Tests for ammonyte.core.rqa_res
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
from ammonyte.core.transitions import DeterministicTransitions

class TestCoreRQAResSmooth:
    '''Tests for smooth function'''

    @pytest.mark.parametrize('block_size',(5,None))
    def test_smooth_t0(self, block_size, gen_series_with_transitions):
        '''Test that smooth block-averages values in place, preserving length'''
        ts_normal = gen_series_with_transitions(add_transitions=False, nt=100)
        amt_td = amt.TimeEmbeddedSeries(ts_normal,3,1)
        rm = amt_td.create_recurrence_matrix(1)
        lp_series = rm.laplacian_eigenmaps(5,3)

        original_length = len(lp_series.value)
        result = lp_series.smooth(block_size)

        # smooth_series modifies and returns the same object
        assert result is lp_series
        assert isinstance(result, amt.RQARes)
        assert len(result.value) == original_length
        assert len(result.time) == original_length
        assert not np.any(np.isnan(result.value))

        # values are block-constant: consecutive blocks share the same value
        effective_block_size = block_size if block_size is not None else int(original_length / 15)
        assert np.allclose(result.value[:effective_block_size], result.value[0])

class TestCoreRQAResConfidenceSmoothPlot:
    '''Tests for confidence_smooth_plot function'''

    def test_confidence_smooth_plot_t0(self, gen_series_with_transitions):
        '''Test confidence_smooth_plot runs without error with default arguments'''
        ts_normal = gen_series_with_transitions(add_transitions=False, nt=100)
        amt_td = amt.TimeEmbeddedSeries(ts_normal, 3, 1)
        rm = amt_td.create_recurrence_matrix(1)
        lp_series = rm.laplacian_eigenmaps(5, 3)
        lp_series.confidence_smooth_plot(block_size=3)
        plt.close('all')

    @pytest.mark.parametrize('transition_interval,ci_kwargs', ([(0, 1), None], [(1, -1), None], [None, {'upper': 75, 'lower': 15}]))
    def test_confidence_smooth_plot_t1(self, transition_interval, ci_kwargs, gen_series_with_transitions):
        '''Test confidence_smooth_plot with different confidence interval options'''
        ts_normal = gen_series_with_transitions(add_transitions=False, nt=100)
        amt_td = amt.TimeEmbeddedSeries(ts_normal, 3, 1)
        rm = amt_td.create_recurrence_matrix(1)
        lp_series = rm.laplacian_eigenmaps(5, 3)
        lp_series.confidence_smooth_plot(block_size=3, transition_interval=transition_interval, ci_kwargs=ci_kwargs)
        plt.close('all')


class TestCoreRQAResLermTransitions:
    '''Tests for lerm_transitions function'''

    def test_lerm_transitions_t0(self, gen_series_with_transitions):
        '''Test lerm_transitions returns a DeterministicTransitions object'''
        ts_normal = gen_series_with_transitions(add_transitions=False, nt=100)
        amt_td = amt.TimeEmbeddedSeries(ts_normal, 3, 1)
        rm = amt_td.create_recurrence_matrix(1)
        lp_series = rm.laplacian_eigenmaps(5, 3)
        smoothed = lp_series.smooth(block_size=3)

        result = smoothed.lerm_transitions(upper=95, lower=5, w=30, n_samples=500)

        assert isinstance(result, DeterministicTransitions)

class TestCoreRQAResConfidenceFillPlot:
    '''Tests for confidence fill plot function'''

    @pytest.mark.parametrize('line_color,fill_color,fill_alpha,legend,label,xlabel,ylabel,title,plot_kwargs,lgd_kwargs',
                            (['green','purple',1,True,'label_test','xlabel_test','ylabel_test','title_test',{'alpha':.5},{'fontsize':42}],
                            [None,None,None,False,None,None,None,None,None,None]))
    def test_confidence_fill_plot_t0(self,line_color,fill_color,fill_alpha,legend,label,xlabel,ylabel,title,plot_kwargs,lgd_kwargs, gen_series_with_transitions):
        '''Testing different visual plot arguments'''
        #Parameter choices are completely arbitrary, just want to test if the plotting function works
        ts_normal = gen_series_with_transitions(add_transitions=False, nt=100)
        amt_td = amt.TimeEmbeddedSeries(ts_normal,3,1)
        rm = amt_td.create_recurrence_matrix(1)
        lp_series = rm.laplacian_eigenmaps(5,3)
        lp_series.confidence_fill_plot(line_color=line_color,fill_color=fill_color,fill_alpha=fill_alpha,
                                       legend=legend,label=label,xlabel=xlabel,ylabel=ylabel,title=title,plot_kwargs=plot_kwargs,
                                       lgd_kwargs=lgd_kwargs )
        plt.close('all')

    @pytest.mark.parametrize('transition_interval,ci_kwargs',([(0,1),None],[(1,-1),None],[None,{'upper':75,'lower':15}]))
    def test_confidence_fill_plot_t1(self,transition_interval,ci_kwargs, gen_series_with_transitions):
        '''Testing different confidence interval calculations'''
        #Parameter choices are completely arbitrary, just want to test if the plotting function works
        ts_normal = gen_series_with_transitions(add_transitions=False, nt=100)
        amt_td = amt.TimeEmbeddedSeries(ts_normal,3,1)
        rm = amt_td.create_recurrence_matrix(1)
        lp_series = rm.laplacian_eigenmaps(5,3)
        lp_series.confidence_fill_plot(transition_interval=transition_interval,ci_kwargs=ci_kwargs)
        plt.close('all')

    @pytest.mark.parametrize('background_kwargs',(None,{'alpha':1}))
    def test_confidence_fill_plot_t2(self,background_kwargs, gen_series_with_transitions):
        '''Testing with background plot'''
        #Parameter choices are completely arbitrary, just want to test if the plotting function works
        ts_normal = gen_series_with_transitions(add_transitions=False, nt=100)
        amt_td = amt.TimeEmbeddedSeries(ts_normal,3,1)
        rm = amt_td.create_recurrence_matrix(1)
        lp_series = rm.laplacian_eigenmaps(5,3)
        lp_series.confidence_fill_plot(background_series=ts_normal,background_kwargs=background_kwargs)
        plt.close('all')


class TestCoreRQAResPlotEigenmaps:
    '''Tests for plot_eigenmaps function'''

    def test_plot_eigenmaps_t0(self, gen_series_with_transitions):
        '''Test that plot_eigenmaps runs without error and returns a figure and axes'''
        ts_normal = gen_series_with_transitions(add_transitions=False, nt=100)
        amt_td = amt.TimeEmbeddedSeries(ts_normal, 3, 1)
        rm = amt_td.create_recurrence_matrix(1)
        lp_series = rm.laplacian_eigenmaps(5, 3)

        # group bounds must fall within lp_series.time and be exact points on ts_normal.time
        groups = [(4, 50), (50, 94)]
        fig, ax = lp_series.plot_eigenmaps(groups=groups, axes=[1, 2])

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close('all')

    def test_plot_eigenmaps_malformed_group_raises_t0(self, gen_series_with_transitions):
        '''Test that a malformed group (not a (start, stop) pair) raises ValueError'''
        ts_normal = gen_series_with_transitions(add_transitions=False, nt=100)
        amt_td = amt.TimeEmbeddedSeries(ts_normal, 3, 1)
        rm = amt_td.create_recurrence_matrix(1)
        lp_series = rm.laplacian_eigenmaps(5, 3)

        with pytest.raises(ValueError):
            lp_series.plot_eigenmaps(groups=[(4, 50), 'bad'], axes=[1, 2])
        plt.close('all')

    def test_plot_eigenmaps_stop_out_of_bounds_raises_t0(self, gen_series_with_transitions):
        '''Test that a stop time outside the RQARes time bounds raises ValueError'''
        ts_normal = gen_series_with_transitions(add_transitions=False, nt=100)
        amt_td = amt.TimeEmbeddedSeries(ts_normal, 3, 1)
        rm = amt_td.create_recurrence_matrix(1)
        lp_series = rm.laplacian_eigenmaps(5, 3)

        with pytest.raises(ValueError):
            lp_series.plot_eigenmaps(groups=[(4, 500)], axes=[1, 2])
        plt.close('all')


class TestCoreRQAResPlotEigenmapsFI:
    '''Tests for plot_eigenmaps_FI function'''

    def test_plot_eigenmaps_fi_t0(self, gen_series_with_transitions):
        '''Test that plot_eigenmaps_FI runs without error and returns a 3D figure and axes'''
        ts_normal = gen_series_with_transitions(add_transitions=False, nt=100)
        amt_td = amt.TimeEmbeddedSeries(ts_normal, 3, 1)
        rm = amt_td.create_recurrence_matrix(1)
        lp_series = rm.laplacian_eigenmaps(5, 3)

        groups = [(4, 50), (50, 94)]
        fig, ax = lp_series.plot_eigenmaps_FI(groups=groups, axes=[1, 2])

        assert isinstance(fig, plt.Figure)
        plt.close('all')

    def test_plot_eigenmaps_fi_block_smooth_false_raises_t0(self, gen_series_with_transitions):
        '''Test that block_smooth=False raises ValueError (not yet implemented, per source)'''
        ts_normal = gen_series_with_transitions(add_transitions=False, nt=100)
        amt_td = amt.TimeEmbeddedSeries(ts_normal, 3, 1)
        rm = amt_td.create_recurrence_matrix(1)
        lp_series = rm.laplacian_eigenmaps(5, 3)

        groups = [(4, 50), (50, 94)]
        with pytest.raises(ValueError):
            lp_series.plot_eigenmaps_FI(groups=groups, axes=[1, 2], block_smooth=False)
        plt.close('all')