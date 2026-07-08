#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ..core.recurrence_matrix import RecurrenceMatrix

class RecurrenceNetwork(RecurrenceMatrix):
    '''Recurrence network object. Used for Recurrence Network Analysis (RNA).

    Child of ammonyte.RecurrenceMatrix, so it has all of the RecurrenceMatrix methods
    plus additional methods defined here
    '''
    def __init__(self,matrix,time,epsilon,series=None,value_name=None,value_unit=None,time_name=None,time_unit=None,label=None):
        '''
        Parameters
        ----------

        matrix : numpy.ndarray
            2D binary adjacency matrix of the recurrence network, where nodes represent
            states in the embedded phase space and edges connect recurrent state pairs
            within epsilon distance.

        time : array-like
            Time axis corresponding to the rows and columns of the matrix.

        epsilon : float
            Radius threshold used to define edges in the network. Note that the embedding
            parameters (m, tau) are not stored here as they are applied upstream via
            TimeEmbeddedSeries.

        series : ammonyte.Series or pyleoclim.Series, optional
            Original time series from which the network was computed.

        value_name : str, optional
            Name of the value variable.

        value_unit : str, optional
            Units of the value variable.

        time_name : str, optional
            Name of the time variable.

        time_unit : str, optional
            Units of the time variable.

        label : str, optional
            Label for the object.

        See Also
        --------

        ammonyte.RecurrenceMatrix

        ammonyte.TimeEmbeddedSeries
        '''
        self.matrix = matrix
        self.time = time
        self.epsilon = epsilon
        self.series = series
        self.value_name = value_name
        self.value_unit = value_unit
        self.time_name = time_name
        self.time_unit = time_unit
        self.label = label