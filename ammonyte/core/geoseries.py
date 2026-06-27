#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pyleoclim as pyleo

from .series import Series


class GeoSeries(Series, pyleo.GeoSeries):
    '''Ammonyte GeoSeries object for geographically referenced time series.

    Child of both ammonyte.Series and pyleoclim.GeoSeries. Inherits all
    Ammonyte transition detection methods (kstest, ruptures, embed, determinism,
    laminarity) from ammonyte.Series, and all geospatial methods (map, dashboard)
    and metadata (lat, lon, elevation, archiveType) from pyleoclim.GeoSeries.

    Parameters
    ----------
    time : list or numpy.array
        Independent variable (t).

    value : list or numpy.array
        Values of the dependent variable (y).

    lat : float
        Latitude N in decimal degrees. Must be in the range [-90, +90].

    lon : float
        Longitude East in decimal degrees. Must be in the range [-180, +360].

    elevation : float, optional
        Elevation in meters above sea level. Negative values indicate depth
        below global mean sea level.

    time_unit : str, optional
        Units for the time vector (e.g., 'years', 'kyr BP').

    time_name : str, optional
        Name of the time vector (e.g., 'Age', 'Time').

    value_name : str, optional
        Name of the value vector (e.g., 'temperature', 'δ¹⁸O').

    value_unit : str, optional
        Units for the value vector (e.g., 'deg C', '‰').

    label : str, optional
        Name of the time series (e.g., 'NGRIP Ice Core').

    archiveType : str, optional
        Climate archive type (e.g., 'GlacierIce', 'MarineSediment',
        'Speleothem', 'LakeSediment').

    See Also
    --------
    ammonyte.Series : Ammonyte series without geolocation.
    pyleoclim.GeoSeries : Pyleoclim GeoSeries with full geospatial API.

    Examples
    --------

    Create a GeoSeries from the NGRIP ice core record:

    >>> import os, ammonyte as amt
    >>> ngrip = amt.Series.from_csv(os.path.join(os.path.dirname(amt.__file__), 'data', 'NGRIP.csv'))
    >>> ngrip_geo = amt.GeoSeries(
    ...     time=ngrip.time,
    ...     value=ngrip.value,
    ...     lat=75.1,
    ...     lon=-42.32,
    ...     time_name=ngrip.time_name,
    ...     time_unit=ngrip.time_unit,
    ...     value_name=ngrip.value_name,
    ...     value_unit=ngrip.value_unit,
    ...     label=ngrip.label,
    ...     archiveType='GlacierIce'
    ... )
    '''
    pass
