''' Tests for ammonyte.core.geoseries
Naming rules:
1. class: Test{filename}{Class}{method} with appropriate camel case
2. function: test_{method}_t{test_id}

Notes on how to test:
0. Make sure [pytest](https://docs.pytest.org) has been installed: `pip install pytest`
1. execute `pytest {directory_path}` in terminal to perform all tests in all testing files inside the specified directory
2. execute `pytest {file_path}` in terminal to perform all tests in the specified file
3. execute `pytest {file_path}::{TestClass}::{test_method}` in terminal to perform a specific test class/method
4. after `pip install pytest-xdist`, one may execute "pytest -n 4" to test in parallel with number of workers specified by `-n`
5. for more details, see https://docs.pytest.org/en/stable/usage.html
'''

import pytest
import pyleoclim as pyleo
import ammonyte as amt
import numpy as np


def gen_geo_normal(loc=0, scale=1, nt=100, lat=75.1, lon=-42.32):
    '''Generate a GeoSeries with Gaussian random data.'''
    t = np.arange(nt)
    np.random.seed(42)
    v = np.random.normal(loc=loc, scale=scale, size=nt)
    return amt.GeoSeries(time=t, value=v, lat=lat, lon=lon)


class TestCoreGeoSeriesInstantiation:
    '''Tests for GeoSeries instantiation and inheritance'''

    def test_is_ammonyte_geoseries_t0(self):
        '''GeoSeries is an instance of ammonyte.GeoSeries'''
        ts = gen_geo_normal()
        assert isinstance(ts, amt.GeoSeries)

    def test_is_pyleoclim_geoseries_t0(self):
        '''GeoSeries is an instance of pyleoclim.GeoSeries'''
        ts = gen_geo_normal()
        assert isinstance(ts, pyleo.GeoSeries)

    def test_is_ammonyte_series_t0(self):
        '''GeoSeries is an instance of ammonyte.Series'''
        ts = gen_geo_normal()
        assert isinstance(ts, amt.Series)

    def test_lat_lon_t0(self):
        '''GeoSeries stores lat and lon correctly'''
        ts = gen_geo_normal(lat=75.1, lon=-42.32)
        assert ts.lat == 75.1
        assert ts.lon == -42.32


class TestCoreGeoSeriesEmbed:
    '''Tests for embed method on GeoSeries'''

    @pytest.mark.parametrize('m,tau', [(10, 5), (10, None)])
    def test_embed_t0(self, m, tau):
        '''embed returns a TimeEmbeddedSeries'''
        ts = gen_geo_normal()
        result = ts.embed(m, tau)
        assert isinstance(result, amt.TimeEmbeddedSeries)


class TestCoreGeoSeriesKstest:
    '''Tests for kstest method on GeoSeries'''

    def test_kstest_returns_transitions_t0(self):
        '''kstest returns a DeterministicTransitions object'''
        ts = gen_geo_normal()
        result = ts.kstest(w_min=5, w_max=20)
        assert isinstance(result, amt.DeterministicTransitions)


class TestCoreGeoSeriesRuptures:
    '''Tests for ruptures method on GeoSeries'''

    def test_ruptures_returns_transitions_t0(self):
        '''ruptures returns a DeterministicTransitions object'''
        ts = gen_geo_normal()
        result = ts.ruptures(algo='Pelt', cost='rbf', pen=10)
        assert isinstance(result, amt.DeterministicTransitions)
