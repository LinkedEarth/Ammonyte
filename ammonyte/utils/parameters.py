import multiprocessing as mp
import itertools
import pyleoclim as pyleo
import numpy as np

from sklearn.feature_selection import mutual_info_regression
from scipy.signal import argrelextrema
from tqdm import tqdm

from ..utils.rm import rm
from ..utils.range_finder import range_finder


__all__ = ['tau_search']

def tau_search(series,num_lags=30,return_MI = False):
    '''Find optimal tau value for time delay embedding.

    First minimum of mutual information between series and time lagged copies of itself
    is "optimal" in this case in accordance with Abarnabel's "Analysis of Observed Chaotic Data"

    Parameters
    ----------

    series : pyleo.Series
        Series for which we'd like to find the optimal tau value

    num_lags : int
        Number of time delays to consider. Default is 30

    return_MI : bool, {True,False}
        Whether or not to return the list of mutual information values.
        Useful if the first minimum seems spurious and you'd like to inspect the results.

    Returns
    -------

    tau : int
        Optimal time delay parameter according to first minimum of mutual information

    MI : list
        List of mutual information values.
        Indices + 1 correspond to amount of lag (index 0 is lag 1, index 1 is lag 2, etc.).
        Only returned if return_MI is set to True.

    Citations
    ---------

    I., Abarbanel Henry D. Analysis of Observed Chaotic Data. Springer, 1997.
    '''
    lags = np.arange(1,num_lags)
    MI = []

    for lag in lags:
        values = series.value[:-lag].reshape(-1, 1)
        lagged_values = series.value[lag:]
        MI.append(mutual_info_regression(values, lagged_values, random_state=42))

    #Make MI an array
    MI = np.array(MI)

    #Find minimum values
    extrema = argrelextrema(MI,np.less)[0]

    try:
        #If extrema can be identified, use that. If not it means that MI probably went to zero, so pick first instance of zero
        if len(extrema)>0:
            best_tau = extrema[0] + 1
        elif 0 in MI:
            best_tau = min(np.where(MI==0)[0]) + 1
    except:
        print(MI)
        raise ValueError()

    if return_MI is True:
        return best_tau,MI
    else:
        return best_tau

