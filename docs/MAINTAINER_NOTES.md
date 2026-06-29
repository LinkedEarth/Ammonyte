# Maintainer Notes — Docs Build Pipeline

This file is not part of the published site (it isn't linked from `_quarto.yml`). It documents
how the doc-build scripts in `docs/scripts/` expect things to be set up, so the conventions don't
have to be rediscovered by reading the scripts each time.

## Writing docstrings that render correctly

`quartodoc` (via `griffe`) parses your package's docstrings to build `docs/api/`. Use
[numpy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html) — `Parameters`,
`Returns`, `Examples`, etc. as `Section\n----------` headers.

### Adding a runnable example figure to a docstring

`docs/scripts/make_doc_figures.py` scans every docstring for a fenced ` ```python ` block that
calls `plt.savefig(...)`, executes that block, and writes the resulting image to `docs/figures/`.
`docs/scripts/inject_doc_figures.py` then inserts a Markdown image reference into the rendered
`docs/api/*.qmd` page, right after the code block.

For this to work, the `plt.savefig(...)` call inside your docstring's `Examples` section must:

1. End the filename in `_example.png` — the prefix before `_example` becomes the figure's name.
2. Use the literal path prefix `docs/reference/figures/` in the string, even though the file is
   actually written to `docs/figures/` and referenced as `../figures/<name>.png` in the rendered
   page. `inject_doc_figures.py`'s regex matches on that literal substring — it is a convention
   marker, not a real path, and changing it will make the figure stop being detected.

Example:

```python
def my_function():
    """One-line summary.

    Examples
    --------
    ```python
    import matplotlib.pyplot as plt
    import ammonyte

    # ... build and plot something ...
    plt.savefig('docs/reference/figures/MyFunction_example.png', dpi=150, bbox_inches='tight')
    ```
    """
```

Run `python scripts/make_doc_figures.py --list` to see every example discovered, or
`--name MyFunction` to regenerate just one figure. Figures are not regenerated automatically on
every render — rerun the script after changing an example's code.

## Hardcoded paths to update if you reorganize notebooks or tutorials

A few paths are hardcoded in the scripts rather than read from `_quarto.yml` or a config file.
If you rename or add notebook subdirectories, or rename the `tutorials/` directory, update these:

- **`docs/scripts/sync_notebooks.py`**: `NOTEBOOK_DIR` (currently `'notebooks'`) and
  `NOTEBOOK_SUBDIRS` (currently `["model_demos", "functionality_demos", "base_classes"]`) — only
  notebooks inside the listed subdirectories get mirrored into `docs/notebooks/` before each
  render. A new subdirectory under `notebooks/` will be silently skipped until it's added here.
- **`docs/scripts/generate_tutorials_index.py`**: the sidebar lookup hardcodes
  `sidebar.get("id") == "tutorials"` and writes its output to
  `DOCS_DIR / "tutorials" / "_tutorials_table.qmd"`. If you rename the `tutorials/` directory or
  its sidebar `id` in `_quarto.yml`, update both of these to match.
- **`docs/scripts/sync_notebooks.py`**: `GITHUB_REPO` and `GITHUB_BRANCH` drive the generated
  "Launch on Binder" / "Download notebook" links — keep these in sync if the repo moves or the
  default branch changes (they are set from the `docsmith` spec at generation time, but won't
  auto-update afterward).
