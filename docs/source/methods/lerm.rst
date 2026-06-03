Laplacian Eigenmaps of Recurrence Matrices (LERM)
==================================================

LERM detects abrupt transitions by tracking changes in the dynamical
structure of a time series rather than its raw values. The method works
through a four-step pipeline that transforms the original series into a
Fisher Information signal, which serves as an indicator of system
instability. Transitions are then detected from that signal.

This method is based on James et al. (2024) [1]_.


Pipeline
--------

**Step 1 — Time delay embedding**

The 1D time series is reconstructed into a multi-dimensional phase space
using delay embedding. This captures the dynamical state of the system at
each time point rather than just its scalar value.

Parameters: ``m`` (embedding dimension), ``tau`` (time delay — computed
automatically from the first minimum of mutual information if not provided)

**Step 2 — Recurrence network**

Points in phase space that fall within a distance ``eps`` of each other
are connected, producing a recurrence network. The algorithm searches for
an ``eps`` value that achieves a target network density.

Parameters: ``eps`` (initial search value), ``target_density`` (desired
fraction of connected pairs), ``tolerance`` (acceptable deviation from
target density), ``amp`` (controls step size during the search)

**Step 3 — Fisher Information**

Fisher Information is computed from the recurrence network using a sliding
window via Laplacian Eigenmaps. High Fisher Information indicates the
system is near a critical transition; low values indicate stability.

Parameters: ``w_size`` (window size in data points), ``w_incre`` (step
size between successive windows)

**Step 4 — Smoothing**

Block averaging is applied to the raw Fisher Information to reduce
high-frequency noise before transition detection.

Parameter: ``block_size`` (number of points averaged per block)


Transition Detection
--------------------

Once the smoothed Fisher Information series is obtained, transitions can
be detected using one of two approaches:

**LERM threshold method**

Detects transitions where Fisher Information crosses between a lower and
upper confidence bound. A complete upward transition occurs when the
signal moves from below the lower bound to above the upper bound; a
downward transition is the reverse. Bounds are computed automatically via
bootstrapping if not provided manually.

Parameters: ``upper`` (upper percentile, default 95), ``lower`` (lower
percentile, default 5), ``w`` (bootstrap sample size), ``n_samples``
(number of bootstrap samples). Alternatively, pass ``transition_interval``
as a (upper_bound, lower_bound) tuple to set bounds manually.

**KS test method**

The augmented KS test (see :doc:`ks_test`) can be applied directly to the
smoothed Fisher Information series as an alternative detection approach.
This treats the Fisher Information as the input signal rather than the
original time series.


Results
-------

**Fisher Information series**

The smoothed Fisher Information is an intermediate output worth inspecting
before running detection. Peaks in the signal correspond to periods of
dynamical instability and are the basis for all transition detections.

**Transitions**

Both detection methods return a ``Transitions`` object with:

``jump_times``
    Times of detected transitions.

``jump_directions``
    ``+1`` for upward (Fisher Information rises through upper bound),
    ``-1`` for downward (Fisher Information falls through lower bound).

Calling ``transitions.plot()`` displays both the Fisher Information and
the original time series with detected transitions marked as vertical
lines: red for upward, blue for downward.


References
----------

.. [1] James, A., Emile-Geay, J., Malik, N., & Khider, D. (2024).
   Detecting Paleoclimate Transitions With Laplacian Eigenmaps of
   Recurrence Matrices (LERM). Paleoceanography and Paleoclimatology.
   https://doi.org/10.1029/2023PA004700
