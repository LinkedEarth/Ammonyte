---
title: 'Ammonyte: A Python Package for Multi-Method Detection of Transitions in Paleoclimate Time Series'
tags:
  - Python
  - paleoclimate
  - tipping points
  - abrupt transitions
  - nonlinear time series analysis
  - changepoint detection
  - Kolmogorov-Smirnov test
  - recurrence analysis
authors:
  - name: Maryam Niati
    orcid: 0009-0001-1523-7989
    affiliation: 1
  - name: Alexander James
    orcid: 0000-0001-8561-3188
    affiliation: 1
  - name: Julien Emile-Geay
    orcid: 0000-0001-5920-4751
    affiliation: 1
  - name: Deborah Khider
    orcid: 0000-0001-7501-8430
    affiliation: 2
affiliations:
  - name: Climate Dynamics Lab, University of Southern California, Los Angeles, CA, USA
    index: 1
  - name: Information Sciences Institute, University of Southern California, Marina del Rey, CA, USA
    index: 2
date: 2026-06-10
bibliography: paper.bib
---

# Summary

Ammonyte is an open-source Python package for detecting abrupt transitions
and tipping points in paleoclimate time series, available at
<https://github.com/LinkedEarth/Ammonyte> [@Niati2026]. Proxy records from
ice cores, marine sediments, and speleothems reveal that Earth's climate has
undergone repeated regime shifts that fundamentally reorganized global climate
patterns [@Alley2003; @Lenton2008]. Identifying when and how these transitions
occurred is essential for understanding past climate variability and assessing
the sensitivity of the modern climate system to future forcing.

Ammonyte provides a unified, paleoclimate-oriented interface to three
methodologically distinct approaches for transition detection: (1) the augmented
Kolmogorov–Smirnov (KS) test [@Bagniewski2021]; (2) optimization-based
changepoint detection via the `ruptures` library [@Truong2020]; and (3)
Laplacian Eigenmaps of Recurrence Matrices (LERM) [@James2024]. The package
extends Pyleoclim [@Khider2022] and inherits its full suite of tools for
preprocessing irregularly sampled, age-uncertain proxy records. Ammonyte can
be installed via `pip install ammonyte`.

# Statement of Need

Paleoclimatologists and climate scientists working with proxy time series
routinely need to identify abrupt transitions and tipping points, but the
methods for doing so are scattered across disciplines, implemented in
different languages, and rarely designed with the particular challenges of
paleoclimate data (irregular sampling, age uncertainty, short and noisy
records) in mind. Detecting these transitions today typically requires
researchers to implement statistical tests from the literature by hand, adapt
general-purpose changepoint packages built for regularly sampled data, or
write custom code to bridge preprocessing, detection, and visualization,
a substantial and error-prone undertaking that falls outside most
paleoclimatologists' core expertise.

Different transition detection methods also rest on fundamentally different
assumptions, so no single algorithm is reliable across all record types. The
augmented KS test [@Bagniewski2021] identifies points where the statistical
distribution of values changes abruptly. Optimization-based segmentation via
`ruptures` [@Truong2020] finds breakpoints by minimizing a cost function,
offering flexible parametric models and multiple search algorithms. LERM
[@James2024] exploits the geometry of the system's reconstructed state space
via recurrence analysis and Laplacian eigenmapping, making it uniquely
sensitive to gradual or dynamical regime changes that may be invisible to
amplitude-based methods.

Each of these methods also has its own advantages and disadvantages. The
augmented KS test is lightweight and does not require much computation time,
but it is prone to overfitting, which tends to produce high recall at the
cost of lower precision — that is, it detects most true transitions but also
flags more false positives that require careful post-hoc filtering. Some
`ruptures` search algorithms are similarly lightweight, but others, such as
dynamic programming search or the RBF cost function, can take considerably
longer depending on the length of the time series; overall, however, the
performance of most `ruptures` methods is reasonable relative to their
computational cost, making them practically useful. LERM, by contrast, is
grounded in system dynamics rather than statistics, so unlike the other two
methods it is independent of the time series' statistical behavior, allowing
it to detect nonlinear regime shifts that purely statistical methods may
miss. Its disadvantage is that it also takes longer to run and can be
computationally heavy.

Ammonyte addresses both problems for the paleoclimate community. First, by
extending Pyleoclim [@Khider2022], it gives all three methods direct access
to age-uncertain, irregularly sampled proxy series and their existing
preprocessing tools, removing the need to reimplement data handling for each
method separately. Second, because each method has distinct strengths and
failure modes, Ammonyte is designed to make applying multiple approaches to
the same record straightforward through a single, consistent interface, so
that cross-validation of results is a natural part of the workflow rather
than a separate engineering effort. This makes rigorous, multi-method
transition detection accessible to paleoclimate researchers without a
background in signal processing or dynamical systems theory.

# State of the Field

Two existing tools address parts of this problem. The `ruptures` library
[@Truong2020] finds breakpoints in any time series by minimizing a cost
function, but is agnostic to paleoclimate data's age uncertainty and
irregular sampling. `TransitionsInTimeseries.jl` [@SwierczekJereczek2024] is
explicitly paleoclimate-aware and provides a sliding-window interface for
indicators such as the augmented KS statistic, permutation entropy, and
critical slowing down, but it is limited to statistical, indicator-based
detection, it has no recurrence-based dynamical method comparable to LERM, and is built in Julia rather than the Python/Pyleoclim ecosystem most
paleoclimatologists use.

No existing package combines distribution-based, optimization-based, and
recurrence-based detection within one paleoclimate-aware interface. By
extending Pyleoclim [@Khider2022], Ammonyte inherits proxy-specific
preprocessing (age-uncertainty propagation, interpolation, binning) and
applies all three detection paradigms to the same series through one API,
letting researchers cross-validate methodologically distinct detectors
without switching languages or tools.

# Software design

Ammonyte extends Pyleoclim's `Series` class rather than exposing a separate
API, so that preprocessing (interpolation, binning, filtering, age
uncertainty propagation) and detection operate on the same object and users
never need to reformat data between steps. The three detection workflows
are:

**Augmented KS test** (`Series.kstest()`): Implements the sliding-window
method of @Bagniewski2021, scanning multiple window sizes with additional
criteria for minimum sample size, rate-of-change, and standard deviation
ratio. Returns transition times, directions, KS D-statistics, and p-values.

**Ruptures-based changepoint detection** (`Series.ruptures()`): Wraps
`ruptures` [@Truong2020] with paleoclimate-appropriate defaults, exposing
six search algorithms and multiple cost functions. Returns breakpoint times
and inferred transition directions.

**LERM** (`RecurrenceMatrix.laplacian_eigenmaps()` → `Series.lerm_transitions()`):
Constructs a time-delay embedding, computes the recurrence matrix via PyRQA
[@Rawald2017], applies Laplacian eigenmapping over sliding windows, computes
Fisher information, and detects transitions as confidence-interval crossings
of the Fisher information signal. Intermediate objects are available for
inspection and visualization.

Despite their different internal statistics, all three methods return a
single `DeterministicTransitions` type carrying transition times,
directions, the originating series, method parameters, and method-specific
statistics — rather than method-specific output formats — so results from
different detectors are directly comparable and plottable, which is what
makes the cross-validation workflow described above practical rather than
a manual reconciliation task. Full documentation and worked examples are
available in the package repository.

# Acknowledgements

This work was supported by NSF grant RISE-2425885.

# References
