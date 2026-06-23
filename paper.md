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

Different transition detection methods rest on fundamentally different
assumptions. The augmented KS test [@Bagniewski2021] identifies points where
the statistical distribution of values changes abruptly. Optimization-based
segmentation via `ruptures` [@Truong2020] finds breakpoints by minimizing a
cost function, offering flexible parametric models and multiple search
algorithms. LERM [@James2024] exploits the geometry of the system's
reconstructed state space via recurrence analysis and Laplacian eigenmapping,
making it uniquely sensitive to gradual or dynamical regime changes that may
be invisible to amplitude-based methods.

Because each method has distinct strengths and failure modes, Ammonyte is
designed to make applying multiple approaches to the same record
straightforward, so that cross-validation of results is a natural part of
the workflow.

# Functionality

Ammonyte extends Pyleoclim's `Series` class so that all preprocessing
capabilities (interpolation, binning, filtering, age uncertainty propagation)
are available before any transition detection step. The three detection
workflows are:

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

All methods return a `DeterministicTransitions` object carrying transition
times, directions, the originating series, method parameters, and
method-specific statistics. Full documentation and worked examples are
available in the package repository.

# Acknowledgements

This work was supported by NSF grant RISE-2425885.

# References
