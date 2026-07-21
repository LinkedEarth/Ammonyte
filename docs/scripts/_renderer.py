"""Custom quartodoc renderer that adds GitHub source links to documentation.

This renderer extends quartodoc's MdRenderer to add [source] links that point
to the corresponding GitHub file and line numbers for functions and classes.

It also upgrades plain, non-doctest ```python``` fences in docstring
Examples sections to executable ```{python}``` cells, so `quarto render`
actually runs them and shows their output (print statements, plots, reprs)
instead of just displaying the source text. Doctest-style (``>>>``) examples
are left as static code display, since they aren't valid standalone Python.

Usage in _quarto.yml (quartodoc build/render must be run with cwd=docs/,
so the module is importable as scripts._renderer):
    quartodoc:
      renderer:
        style: scripts._renderer.py
        repo_url: https://github.com/LinkedEarth/Ammonyte
        branch: main
        source_dir: ammonyte
"""

from __future__ import annotations

import re
from typing import Union

from plum import dispatch
from quartodoc import ast as qast
from quartodoc import layout
from quartodoc.renderers import MdRenderer

# Matches a fenced ```python block within a block of markdown text, capturing
# its body (used to upgrade plain example fences to executable {python} cells).
_PYTHON_FENCE_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)


class SourceLinkMdRenderer(MdRenderer):
    """Markdown renderer with GitHub source links.

    Adds [source] links to function and class documentation that link
    to the corresponding file and line numbers on GitHub.

    Parameters
    ----------
    repo_url : str
        The base URL of the GitHub repository.
    branch : str
        The branch name to link to (default: "main").
    source_dir : str
        The directory containing source code (default: "src").
    **kwargs
        Additional arguments passed to MdRenderer.
    """

    style = "source-links"

    def __init__(
        self,
        repo_url: str = "https://github.com/michaelaye/planetarypy",
        branch: str = "main",
        source_dir: str = "src",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.repo_url = repo_url.rstrip("/")
        self.branch = branch
        self.source_dir = source_dir

    # executable examples ----

    @staticmethod
    def _is_doctest(code: str) -> bool:
        """True if *code* is a doctest-style (``>>>``) snippet.

        Doctest snippets aren't valid Python on their own (the ``>>>`` /
        ``...`` prompts are part of the text), so they must stay as plain,
        non-executed code fences.
        """
        return ">>>" in code

    @dispatch
    def render(self, el: qast.ExampleCode):
        """Render a docstring Examples block classified entirely as code.

        Plain runnable snippets become executable ```{python}``` cells so
        Quarto actually runs them and shows their output (print, plots,
        reprs). Doctest-style snippets fall back to the default static
        rendering, since ``>>>`` prompts aren't executable as-is.
        """
        if self._is_doctest(el.value):
            return super().render(el)
        return f"""```{{python}}\n{el.value}\n```"""

    @dispatch
    def render(self, el: qast.ExampleText):
        """Render a docstring Examples block that mixes prose and code.

        Numpydoc-style Examples sections that use plain fenced code blocks
        (rather than ``>>>`` doctests) are parsed by griffe as a single text
        blob rather than separate code/text items. Upgrade each embedded
        ```python``` fence that isn't a doctest snippet to an executable
        ```{python}``` cell in place, leaving surrounding prose untouched.
        """

        def _upgrade(match: re.Match) -> str:
            code = match.group(1)
            if self._is_doctest(code):
                return match.group(0)
            return f"```{{python}}\n{code}```"

        return _PYTHON_FENCE_RE.sub(_upgrade, el.value)

    def _get_source_link(self, obj) -> str:
        """Generate a GitHub source link for a griffe object.

        Parameters
        ----------
        obj
            A griffe object (Function, Class, Module, etc.)

        Returns
        -------
        str
            Markdown link to the source on GitHub, or empty string if
            line number info is unavailable.
        """
        # Handle Alias objects by getting the target
        if hasattr(obj, "target") and obj.target is not None:
            obj = obj.target

        # Get line number - required for source link
        lineno = getattr(obj, "lineno", None)
        if lineno is None:
            return ""

        # Get the file path relative to the package
        # griffe provides relative_package_filepath which is like "planetarypy/config.py"
        rel_path = getattr(obj, "relative_package_filepath", None)
        if rel_path is None:
            # Fallback: try to get filepath and extract relative part
            filepath = getattr(obj, "filepath", None)
            if filepath is None:
                return ""
            # Try to find the package in the path
            filepath_str = str(filepath)
            if "planetarypy" in filepath_str:
                idx = filepath_str.find("planetarypy")
                rel_path = filepath_str[idx:]
            else:
                return ""

        # Build the full path: src/planetarypy/module.py (or just rel_path
        # when the package lives at the repo root, e.g. source_dir="").
        full_path = f"{self.source_dir}/{rel_path}" if self.source_dir else rel_path

        # Build the URL with line numbers
        url = f"{self.repo_url}/blob/{self.branch}/{full_path}#L{lineno}"

        # Add end line if available for multi-line definitions
        endlineno = getattr(obj, "endlineno", None)
        if endlineno is not None and endlineno != lineno:
            url += f"-L{endlineno}"

        return f'\n\n[[source]]({url}){{.source-link target="_blank"}}'

    @dispatch
    def render(self, el: Union[layout.DocFunction, layout.DocAttribute]):
        """Render function/attribute documentation with source link."""
        # Get the base rendering from parent class
        base_render = super().render(el)

        # Add source link
        source_link = self._get_source_link(el.obj)

        return base_render + source_link

    @dispatch
    def render(self, el: Union[layout.DocClass, layout.DocModule]):
        """Render class/module documentation with source link."""
        # Get the base rendering from parent class
        base_render = super().render(el)

        # Add source link at the end of the class/module header section
        source_link = self._get_source_link(el.obj)

        if source_link:
            # Insert source link after the signature but before members
            # Find the first double newline after the signature
            parts = base_render.split("\n\n", 3)
            if len(parts) >= 3:
                # Insert after title and signature
                return parts[0] + "\n\n" + parts[1] + source_link + "\n\n" + "\n\n".join(parts[2:])

        return base_render + source_link


# Export as Renderer for quartodoc to discover
Renderer = SourceLinkMdRenderer