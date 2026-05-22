import pyleoclim as pyleo
import numpy as np
import pandas as pd

from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RPComputation
    
def rm(series, eps, m, delay):
    '''Function to calculate recurrence matrix from pyleoclim or pandas series
    
    Parameters
    ----------
    
    series : pyleoclim.Series or pandas.Series
        Timeseries from which the recurrence matrix will be calculated
    eps : float
        Epsilon value for calculation of the matrix.
    m : int
        Embedding parameter
    delay : int
        Delay parameter for time embedding.

    Returns
    -------

    dict
        Dictionary with the following keys:

        - ``'rm'`` : numpy.ndarray — the binary recurrence matrix.
        - ``'time_axis'`` : array-like — the time axis after accounting for embedding.
        - ``'eps'`` : float — the epsilon value used.
    '''
    
    values = series.value
    time_axis = series.time[:-(m-1)*delay]
        
    ts = TimeSeries(values,
                    embedding_dimension = m,
                    time_delay=delay)

    settings = Settings(ts,
                        analysis_type=Classic,
                        neighbourhood=FixedRadius(eps),
                        similarity_measure=EuclideanMetric)

    computation = RPComputation.create(settings,
                                        verbose=False)

    result = computation.run()

    matrix = result.recurrence_matrix

    return {'rm':matrix, 'time_axis':time_axis,'eps':eps}