Augmented Kolmogorov-Smirnov Test
==================================

The augmented Kolmogorov-Smirnov (KS) test detects abrupt transitions in a
time series by comparing the statistical distributions of two adjacent sliding
windows. At each candidate time point, the algorithm places one window
immediately before and one immediately after, then measures the maximum
difference between their empirical cumulative distribution functions — this is
the KS D-statistic.

Where the standard KS test stops at the D-statistic, the augmented version
requires three further conditions before accepting a transition:

- The mean shift between windows must exceed a multiple of each window's
  standard deviation (signal-to-noise condition)
- The absolute difference between window means must exceed a minimum magnitude
  threshold (physical significance condition)
- Each window must contain a minimum number of data points (statistical
  robustness condition)

This multi-condition approach reduces false positives from noisy or
statistically marginal changes. The algorithm repeats this process across
multiple window sizes, and only transitions that are consistent across scales
are retained, making detection robust to the choice of any single window size.

This method is based on Bagniewski et al. (2021) [1]_.


Parameters
----------

**Window configuration**

``w_min``
    Minimum window size in the same time units as the series. Controls the
    shortest timescale transition the method can detect. Must be large enough
    relative to your data's sampling resolution so that each window contains
    sufficient data points.

``w_max``
    Maximum window size. Controls the longest timescale transition the method
    is sensitive to. Increasing this allows detection of broader regime shifts
    but reduces temporal precision in locating the transition.

``n_w``
    Number of window sizes tested, logarithmically spaced between ``w_min``
    and ``w_max``. Higher values provide more scale coverage and more robust
    detection at the cost of computation time.

**Detection thresholds**

``d_c``
    Threshold for the KS D-statistic (range 0–1). The primary sensitivity
    control: lower values detect more transitions, higher values accept only
    pronounced distributional differences. Increasing ``d_c`` reduces false
    positives but may miss subtle transitions.

``s_c``
    The mean shift between windows must exceed ``s_c`` times the standard
    deviation of each window individually. This filters changes that are large
    in absolute terms but small relative to local variability. Increase
    ``s_c`` when working with noisy data.

``x_c``
    Minimum absolute difference between window means, in the same units as
    your data. Prevents physically negligible changes from being flagged even
    if they are statistically significant. If set to ``None``, an automatic
    value is calculated from the data's range and standard deviation.

``n_c``
    Minimum number of data points required in each window half. Ensures the
    KS statistic is estimated from enough samples to be reliable. Transitions
    in sparse regions of the time series are excluded if this condition is
    not met.


Results
-------

Calling ``series.kstest()`` returns a ``Transitions`` object with the
following attributes:

``jump_times``
    Times of detected transitions, sorted chronologically.

``jump_directions``
    Direction of each transition: ``+1`` for upward (mean increases after
    the transition) and ``-1`` for downward (mean decreases).

``d_statistics``
    KS D-statistic for each detected transition. Values closer to 1 indicate
    that the two windows have highly separated distributions, reflecting a
    more pronounced transition. Values near 0 indicate similar distributions.

``p_values``
    P-value associated with each transition. Represents the probability of
    observing a D-statistic this large by chance if the two windows were
    drawn from the same distribution. Lower p-values indicate stronger
    statistical evidence for a real transition.

Calling ``transitions.plot()`` displays the original time series with detected
transitions marked as vertical lines: red for upward transitions and blue for
downward transitions.


References
----------

.. [1] Bagniewski, W., et al. (2021). Automatic detection of abrupt
   transitions in paleoclimate records. *Chaos*, 31(11), 113129.
   https://doi.org/10.1063/5.0062543
