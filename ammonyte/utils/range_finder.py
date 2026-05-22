import numpy as np

def range_finder(eps, density, target_density, tolerance, num_processes, amp):
    '''Compute a range of epsilon values to search for a target recurrence density

    Used internally by TimeEmbeddedSeries.find_epsilon during parallel search.
    Returns either a range of epsilon candidates to evaluate, or signals convergence
    when the current density is within tolerance of the target.

    Parameters
    ----------

    eps : float
        Current epsilon value.

    density : float
        Current recurrence matrix density corresponding to eps.

    target_density : float
        Desired recurrence matrix density.

    tolerance : float
        Acceptable deviation from target_density.

    num_processes : int
        Number of epsilon values to generate in the search range.

    amp : int or float
        Amplitude scaling factor controlling the width of the search range.
        Higher values cover more ground per iteration but converge more slowly.

    Returns
    -------

    eps_range : numpy.ndarray or float
        If not yet converged: array of epsilon values to evaluate next.
        If converged (density within tolerance): the current eps value.

    flag : bool
        True if current density is within tolerance of target (convergence reached),
        False otherwise.

    See Also
    --------

    ammonyte.TimeEmbeddedSeries.find_epsilon
    '''

    if density < (target_density - tolerance):
        
        miss = target_density - density
        eps_bounds = (eps, eps + miss*amp)
        eps_range = np.linspace(min(eps_bounds), max(eps_bounds), num_processes)
        flag = False

        return eps_range, flag
    
    elif density > (target_density + tolerance):
        
        miss = density - target_density
        eps_bounds = (eps, eps - miss*amp)    
        eps_range = np.linspace(min(eps_bounds), max(eps_bounds), num_processes)      
        flag = False
        
        return eps_range, flag
        
    else:
            
        flag = True
        
        return eps, flag