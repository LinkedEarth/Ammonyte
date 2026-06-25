"""Generate static figures for the package reference documentation.

Dynamically discovers every fenced ``python`` code block in the
package docstrings (via griffe), executes those that contain a
``plt.savefig(...)`` call, and writes the resulting figures into
``docs/api/figures/``.

The ``plt.savefig`` call is monkey-patched so the path written in the
docstring is irrelevant — the figure is always saved to ``FIGURE_DIR``
under the bare filename.

This script is the single source of truth for documentation figures: the
examples in the docstrings *are* the figure code, so the rendered images
always match the published documentation.

Run from any directory — all paths are resolved relative to this script.

Usage
-----
Generate all figures::

    python scripts/make_doc_figures.py

Generate a single figure by name::

    python scripts/make_doc_figures.py --name Lorenz63

List discovered figure names::

    python scripts/make_doc_figures.py --list
"""
# Agg backend must be set before any other matplotlib import.
import matplotlib
matplotlib.use('Agg')

import argparse
import re
import sys
import textwrap
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import griffe

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR  = Path(__file__).resolve().parent
DOCS_DIR    = SCRIPT_DIR.parent          # docs/
PROJECT_DIR = DOCS_DIR.parent            # project root
FIGURE_DIR  = DOCS_DIR  / 'figures'

# Ensure the package is importable from exec'd example code.
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Matches a fenced Python code block and captures the body.
FENCE_RE = re.compile(r'```python\s*\n(.*?)```', re.DOTALL)

# Extracts the first positional argument (the path) from plt.savefig(…).
SAVEFIG_RE = re.compile(r"plt\.savefig\(\s*['\"]([^'\"]+)['\"]")

# ---------------------------------------------------------------------------
# plt.savefig redirect
# ---------------------------------------------------------------------------

_real_savefig = plt.savefig


def _redirect_savefig(path, **kwargs):
    """Write the figure to FIGURE_DIR regardless of the path in the docstring."""
    dest = FIGURE_DIR / Path(path).name
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    _real_savefig(str(dest), **kwargs)
    print(f'      → {dest.relative_to(DOCS_DIR)}')


# ---------------------------------------------------------------------------
# Griffe traversal
# ---------------------------------------------------------------------------

def _iter_objects(obj, _seen=None):
    """Yield every griffe object that carries a docstring, depth-first."""
    if _seen is None:
        _seen = set()
    oid = id(obj)
    if oid in _seen:
        return
    _seen.add(oid)

    try:
        docstring = obj.docstring
    except griffe.AliasResolutionError:
        return          # skip unresolvable external aliases (e.g. plt)

    if docstring:
        yield obj

    for member in getattr(obj, 'members', {}).values():
        yield from _iter_objects(member, _seen)


def collect_examples(package: str = 'ammonyte') -> dict[str, str]:
    """Return ``{figure_stem: code_block}`` for every example with savefig.

    Loads *package* statically with griffe, walks all objects, and finds
    fenced ``python`` blocks that contain a ``plt.savefig`` call.  The
    figure stem is derived from the savefig filename (e.g. ``Lorenz63``
    from ``Lorenz63_example.png``).

    Duplicates arising from re-exported aliases are silently collapsed:
    the first encountered code block for each stem wins.

    Parameters
    ----------
    package : str
        Importable package name to scan. Default: the templated package slug.

    Returns
    -------
    dict[str, str]
        Mapping of figure stem → dedented code block string.
    """
    pkg = griffe.load(package, search_paths=[str(PROJECT_DIR)])
    examples: dict[str, str] = {}

    for obj in _iter_objects(pkg):
        raw = obj.docstring.value
        for fence_match in FENCE_RE.finditer(raw):
            code = fence_match.group(1)
            sf_match = SAVEFIG_RE.search(code)
            if not sf_match:
                continue

            # Derive a short name from the filename (strip dir + _example suffix).
            fig_path = sf_match.group(1)
            stem = Path(fig_path).stem                    # e.g. 'Lorenz63_example'
            stem = re.sub(r'_example$', '', stem)         # → 'Lorenz63'

            # Deduplicate: alias re-exports cause the same code to appear several
            # times during traversal; keep only the first occurrence.
            if stem in examples:
                continue

            examples[stem] = textwrap.dedent(code)

    return examples


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def run_example(name: str, code: str) -> bool:
    """Execute *code* with plt.savefig redirected to FIGURE_DIR.

    Parameters
    ----------
    name : str
        Label used in error messages (typically the figure stem).
    code : str
        Python source to execute.

    Returns
    -------
    bool
        ``True`` on success, ``False`` if any exception was raised.
    """
    # Patch at the module level so exec'd `import matplotlib.pyplot as plt`
    # picks up the redirect from the already-cached sys.modules entry.
    matplotlib.pyplot.savefig = _redirect_savefig
    try:
        exec(compile(code, f'<docstring:{name}>', 'exec'),   # noqa: S102
             {'__name__': '__main__', '__builtins__': __builtins__})
        return True
    except Exception:
        traceback.print_exc()
        return False
    finally:
        matplotlib.pyplot.savefig = _real_savefig
        plt.close('all')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--name', metavar='NAME',
        help='generate a single figure (use --list to see available names)',
    )
    parser.add_argument(
        '--list', action='store_true',
        help='list discovered figure names and exit',
    )
    args = parser.parse_args()

    print('Scanning docstrings with griffe…')
    all_examples = collect_examples('ammonyte')

    if not all_examples:
        print('No plt.savefig examples found in docstrings.')
        sys.exit(0)

    if args.list:
        for stem in sorted(all_examples):
            print(f'  {stem}')
        return

    if args.name:
        if args.name not in all_examples:
            print(
                f'Unknown name: {args.name!r}. '
                f'Use --list to see available names.'
            )
            sys.exit(1)
        targets = {args.name: all_examples[args.name]}
    else:
        targets = all_examples

    errors: list[str] = []
    for stem in sorted(targets):
        print(f'  {stem}')
        ok = run_example(stem, targets[stem])
        if not ok:
            errors.append(stem)

    total = len(targets)
    done  = total - len(errors)
    print(f'\nDone. {done}/{total} figure(s) generated.')
    if errors:
        print('Failed:')
        for e in errors:
            print(f'  {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
