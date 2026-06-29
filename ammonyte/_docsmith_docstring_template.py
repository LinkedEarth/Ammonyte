"""Bare numpy-style docstring skeleton to copy-paste for new functions/classes.

Delete this module once Ammonyte has its own documented code —
it's scaffolding, not a real part of the package. See ``docs/MAINTAINER_NOTES.md`` for
the full docstring/figure convention this is based on.
"""


def docsmith_docstring_template(param_one, param_two=None):
    """One-line summary of what this function does.

    Longer description, if needed, goes here.

    Parameters
    ----------
    param_one : type
        What this parameter is for.
    param_two : type, optional
        What this parameter is for. Default is None.

    Returns
    -------
    type
        What gets returned.

    Examples
    --------
    Add a fenced ```python block here with a runnable example. If it ends with
    ``plt.savefig('docs/reference/figures/<Name>_example.png', ...)``, running
    ``python scripts/make_doc_figures.py`` will turn it into a rendered figure
    on this function's reference page — see docs/MAINTAINER_NOTES.md.
    """
    raise NotImplementedError("This is a docstring template, not a real function.")
