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


class TestCoreGeoSeriesInstantiation:
    '''Tests for GeoSeries instantiation and inheritance'''

    def test_is_ammonyte_geoseries_t0(self, gen_geoseries_with_transitions):
        '''GeoSeries is an instance of ammonyte.GeoSeries'''
        ts = gen_geoseries_with_transitions()
        assert isinstance(ts, amt.GeoSeries)

    def test_is_pyleoclim_geoseries_t0(self, gen_geoseries_with_transitions):
        '''GeoSeries is an instance of pyleoclim.GeoSeries'''
        ts = gen_geoseries_with_transitions()
        assert isinstance(ts, pyleo.GeoSeries)

    def test_is_ammonyte_series_t0(self, gen_geoseries_with_transitions):
        '''GeoSeries is an instance of ammonyte.Series'''
        ts = gen_geoseries_with_transitions()
        assert isinstance(ts, amt.Series)

    def test_lat_lon_t0(self, gen_geoseries_with_transitions):
        '''GeoSeries stores lat and lon correctly'''
        ts = gen_geoseries_with_transitions(lat=75.1, lon=-42.32)
        assert ts.lat == 75.1
        assert ts.lon == -42.32


class TestCoreGeoSeriesEmbed:
    '''Tests for embed method on GeoSeries'''

    def test_embed_t0(self, gen_geoseries_with_transitions):
        '''embed returns a TimeEmbeddedSeries'''
        ts = gen_geoseries_with_transitions()
        assert ts.is_evenly_spaced()
        result = ts.embed(10, 5)
        assert isinstance(result, amt.TimeEmbeddedSeries)
