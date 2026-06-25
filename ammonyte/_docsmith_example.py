"""Scaffold-generated example showing the docstring convention quartodoc/griffe expect.

This module is safe to delete once Ammonyte has real documented
functions or classes of its own — it exists only so that running ``quartodoc build`` on a
freshly scaffolded package renders at least one real reference page. See
``docs/MAINTAINER_NOTES.md`` for the full convention (in particular, why the
``plt.savefig`` path below must keep the literal ``docs/reference/figures/`` prefix).
"""

import matplotlib.pyplot as plt
import numpy as np


def docsmith_example_plot(n: int = 50) -> "tuple[np.ndarray, np.ndarray]":
    """Plot a damped sine wave, as a worked example of the docstring convention.

    Parameters
    ----------
    n : int
        Number of points to sample.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The ``(x, y)`` arrays that were plotted.

    Examples
    --------
    ```python
    import matplotlib.pyplot as plt
    import numpy as np
    from ammonyte._docsmith_example import docsmith_example_plot

    x, y = docsmith_example_plot()
    plt.savefig('docs/reference/figures/docsmith_example_plot_example.png', dpi=150, bbox_inches='tight')
    ```
    """
    x = np.linspace(0, 4 * np.pi, n)
    y = np.exp(-x / 8) * np.sin(x)
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    return x, y
