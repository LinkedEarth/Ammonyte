[![PyPI version](https://badge.fury.io/py/ammonyte.svg)](https://badge.fury.io/py/ammonyte)
[![PyPI](https://img.shields.io/badge/python-3.8+-yellow.svg)]()
[![license](https://img.shields.io/github/license/linkedearth/ammonyte.svg)](https://github.com/LinkedEarth/Ammonyte/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/450291131.svg)](https://zenodo.org/badge/latestdoi/450291131)
[![PyPI Downloads](https://static.pepy.tech/badge/ammonyte)](https://pepy.tech/projects/ammonyte)

# Ammonyte

A Python package for multi-method detection of transitions in paleoclimate time series. Developed by [Maryam Niati](https://orcid.org/0009-0001-1523-7989), [Alexander James](https://orcid.org/0000-0001-8561-3188), [Julien Emile-Geay](https://orcid.org/0000-0001-5920-4751), and [Deborah Khider](https://orcid.org/0000-0001-7501-8430) at the [University of Southern California](https://climdyn.usc.edu/).

## Overview

Proxy records from ice cores, marine sediments, and speleothems reveal that Earth's climate has undergone repeated regime shifts that fundamentally reorganized global climate patterns. Identifying when and how these transitions occurred is essential for understanding past climate variability and assessing the sensitivity of the modern climate system to future forcing.

Ammonyte provides a unified, paleoclimate-oriented interface to three methodologically distinct approaches for transition detection, making cross-validation of results across methods a natural part of the workflow. It extends [Pyleoclim](https://github.com/LinkedEarth/Pyleoclim_util) and inherits its full suite of tools for preprocessing irregularly sampled, age-uncertain proxy records.

## Methods

- **Augmented Kolmogorov–Smirnov (KS) Test** — identifies points where the statistical distribution of values changes abruptly using a sliding-window approach
- **Optimization-based Changepoint Detection (Ruptures)** — finds breakpoints by minimizing a cost function across multiple search algorithms
- **Laplacian Eigenmaps of Recurrence Matrices (LERM)** — exploits the geometry of the system's reconstructed state space via recurrence analysis, uniquely sensitive to gradual or dynamical regime changes

## Installation

```bash
pip install ammonyte
```

It is recommended to install inside a dedicated environment:

```bash
conda create -n ammonyte_env python=3.10
conda activate ammonyte_env
pip install ammonyte
```

## Documentation

Full documentation is available at: [linked.earth/Ammonyte](https://linked.earth/Ammonyte/)

- [Installation Guide](https://linked.earth/Ammonyte/latest/get-started/installation.html)
- [API Reference](https://linked.earth/Ammonyte/latest/api/)
- [Method Descriptions](https://linked.earth/Ammonyte/latest/methods/)
- [Tutorials](https://linked.earth/Ammonyte/latest/tutorials/)

## Usage

A minimal example using the Augmented KS Test on the NGRIP ice core record included with the package:

```python
import os
import ammonyte as amt

# Load the NGRIP ice core record
ngrip = amt.Series.from_csv(os.path.join(os.path.dirname(amt.__file__), 'data', 'NGRIP.csv'))

# Detect transitions with the Augmented KS Test
transitions = ngrip.kstest(w_min=0.12, w_max=2.5, n_w=15, d_c=0.77, n_c=3, s_c=2, x_c=0.8)

# Access transition statistics
print(f"D-statistics: {transitions.d_statistics}")
print(f"P-values: {transitions.p_values}")

# Plot the results
transitions.plot()
```

See the [Tutorials](https://linked.earth/Ammonyte/latest/tutorials/) for worked examples of the other two methods (Ruptures, LERM).

## Versions

See the [releases page](https://github.com/LinkedEarth/Ammonyte/releases) for details on what's included in each version.

## Citation

If you use Ammonyte in your research, please cite it using the metadata in [`CITATION.cff`](CITATION.cff):

> Niati, M., James, A., Emile-Geay, J., & Khider, D. (2026). Ammonyte: A Python Package for Multi-Method Detection of Transitions in Paleoclimate Time Series (v1.0.0). https://github.com/LinkedEarth/Ammonyte

## License

This project is licensed under the GNU General Public License v3.0 (or later) (see [`LICENSE`](LICENSE) for details.)

## Contributing

Contributions, bug reports, and feature requests are welcome. Please submit them via [GitHub Issues](https://github.com/LinkedEarth/Ammonyte/issues).
