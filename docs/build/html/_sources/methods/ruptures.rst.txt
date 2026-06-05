Ruptures
========

Ammonyte provides change-point detection through a wrapper around the
`ruptures <https://centre-borelli.github.io/ruptures-docs/>`_ library.
The method finds breakpoints by minimizing the sum of a cost function
across all segments simultaneously, balancing fit quality against the
number of breakpoints through a penalty term.


Algorithms
----------

Six search algorithms are available. The choice of algorithm determines
how the search for breakpoints is conducted and which stopping criterion
is required.

``Pelt``
    Exact detection using dynamic programming with pruning. Efficient for
    long series. Requires ``pen``.

``Binseg``
    Greedy sequential search — recursively splits the series at the most
    significant breakpoint. Requires ``pen`` or ``n_bkps``.

``BottomUp``
    Starts with many breakpoints and iteratively merges adjacent segments.
    Requires ``pen`` or ``n_bkps``.

``Dynp``
    Exact detection via dynamic programming. Guaranteed optimal solution
    but slow on large datasets. Requires ``n_bkps``.

``Window``
    Slides a window across the series and compares statistics on either
    side. Requires ``pen`` or ``n_bkps``, and ``width``.

``KernelCPD``
    Kernel-based detection that captures nonlinear distribution changes
    without parametric assumptions. Only supports ``rbf``, ``linear``, or
    ``cosine`` as the cost function. Requires ``pen`` or ``n_bkps``.


Cost Functions
--------------

The cost function measures how different two segments are from each
other. Choose based on the type of change you expect in your data.

*Mean-based*

``l1``
    Detects shifts in the mean using absolute deviations.

``l2``
    Detects shifts in the mean using squared deviations.

*Distribution-based*

``normal``
    Detects changes in both mean and variance. Assumes Gaussian data.

``rbf``
    Detects all distribution changes — mean, variance, and shape.
    Recommended default for most use cases.

``rank``
    Non-parametric. Detects distribution changes using rank statistics.
    Robust to outliers.

*Trend-based*

``linear``
    Detects changes in linear trend (slope changes).

``clinear``
    Detects piecewise linear trends with continuous connections between
    segments.

*Specialized*

``ar``
    Detects changes in autoregressive model coefficients. Suited for
    temporally dependent data.

``mahalanobis``
    Detects changes in multivariate mean and covariance.

``cosine``
    Detects changes in the orientation of multivariate signals.


Parameters
----------

``algo``
    Search algorithm to use. Default is ``'Pelt'``. See Algorithms
    section for available options and their requirements.

``cost``
    Cost function to use. Default is ``'rbf'``. See Cost Functions
    section for available options.

``pen``
    Penalty parameter. Controls the trade-off between fit quality and
    the number of detected breakpoints. Higher values produce fewer
    breakpoints; lower values produce more. Cannot be used together
    with ``n_bkps``. Required for Pelt; optional for Binseg, BottomUp,
    Window, and KernelCPD.

``n_bkps``
    Exact number of breakpoints to detect. Use when the expected number
    of transitions is known in advance. Cannot be used together with
    ``pen``. Required for Dynp; optional for Binseg, BottomUp, Window,
    and KernelCPD.

``min_size``
    Minimum number of data points between two consecutive breakpoints.
    Default is 2.

``jump``
    Subsampling factor for candidate breakpoint positions. A value of 1
    considers every data point; higher values skip points to speed up
    computation. Default is 1.

``width``
    Window size in data points. Required only for the Window algorithm.


Results
-------

Calling ``series.ruptures()`` returns a ``Transitions`` object with:

``jump_times``
    Times of detected transitions, sorted chronologically.

``jump_directions``
    ``+1`` for upward transitions (mean increases after the breakpoint),
    ``-1`` for downward transitions (mean decreases).

Calling ``transitions.plot()`` displays the original time series with
detected transitions marked as vertical lines: red for upward and blue
for downward.


References
----------

.. [1] Truong, C., Oudre, L., & Vayatis, N. (2020). Selective review of
   offline change point detection methods. *Signal Processing*, 167,
   107299. https://doi.org/10.1016/j.sigpro.2019.107299
