import numpy as np
from scipy.stats import scoreatpercentile
    
def confidence_interval(series,upper=95,lower=5,w=50,n_samples=10000,random_state = 42):
    '''Calculate bootstrap confidence interval bounds for a series

    Designed to compute transition thresholds for Fisher Information series
    via bootstrapping.

    Parameters
    ----------

    series : ammonyte.RQARes or pyleoclim.Series
        Series whose values will be bootstrapped.

    upper : int, optional
        Upper percentile for the confidence interval. Default is 95.

    lower : int, optional
        Lower percentile for the confidence interval. Default is 5.

    w : int, optional
        Bootstrap sample size (number of values drawn per sample). Default is 50.

    n_samples : int, optional
        Number of bootstrap samples to draw. Default is 10000.

    random_state : int, optional
        Seed for the random number generator for reproducibility. Default is 42.

    Returns
    -------

    upper_val : float
        Upper bound of the confidence interval.

    lower_val : float
        Lower bound of the confidence interval.
    '''
    
    rng = np.random.RandomState(seed=random_state)

    sub_arrays = []
    values = list(series.value)
    
    for i in range(n_samples):
        subset = rng.choice(values,size = w,)
        sub_arrays.append(subset)
    
    means = []

    for sequence in sub_arrays:
        means.append(np.mean(sequence))
    
    upper_val = scoreatpercentile(means, upper)
    lower_val = scoreatpercentile(means, lower)
    
    return upper_val, lower_val