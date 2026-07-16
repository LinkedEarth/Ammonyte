"""Pre-render script: copy notebooks from repo root into docs/notebooks/.

Quarto's project root is docs/, so it cannot resolve paths that escape upward
via ../. This script mirrors the top-level notebooks/ tree into docs/notebooks/
before each render so _quarto.yml can reference them with plain relative paths.

If the first cell is a markdown metadata cell (starts with a `# Title` line and
contains Author/Date/Description/Abstract/Keywords fields), it is converted to a
raw YAML front-matter cell in the docs copy so Quarto picks up proper metadata.
The source notebook is never modified.

Only notebooks listed in the INCLUDE sets below are copied; add new ones here
as they are created.
"""

import re
import json
import shutil
import copy
from pathlib import Path

# Repo information
GITHUB_REPO = "LinkedEarth/Ammonyte"
GITHUB_BRANCH = "main"

# Paths relative to this script's location (docs/scripts/)
DOCS_DIR = Path(__file__).parent.parent          # docs/
REPO_ROOT = DOCS_DIR.parent                      # repo root
NOTEBOOK_DIR = 'notebooks'
SRC = REPO_ROOT / NOTEBOOK_DIR
DST = DOCS_DIR / NOTEBOOK_DIR

# Subdirectories to mirror (empty — notebooks live at the root of notebooks/)
NOTEBOOK_SUBDIRS = []

def _parse_markdown_frontmatter(source: list[str]) -> dict | None:
    """Parse a markdown metadata cell into a dict of front-matter fields.

    Expected format:
        # Title
        **Author:** value
        **Date:** value
        _Description:_ value
        _Abstract_
        > prose text (may span multiple > lines)
        **Keywords:** value

    Returns None if the cell doesn't match the expected structure.
    """
    text = "".join(source).strip()

    title_m = re.match(r"^#\s+(.+)", text)
    if not title_m:
        return None

    fields = {"title": title_m.group(1).strip()}

    for key, pattern in [
        ("author",      r"\*\*Author:\*\*\s*(.+)"),
        ("date",        r"\*\*Date:\*\*\s*(.+)"),
        ("description", r"_Description:_\s*(.+)"),
        ("keywords",    r"\*\*Keywords:\*\*\s*(.+)"),
    ]:
        m = re.search(pattern, text)
        if m:
            fields[key] = m.group(1).strip()

    # Abstract: everything between _Abstract_ and the Keywords field (or end of text).
    # Anchor the end on the literal word "Keywords" so minor formatting variation is tolerated.
    # The first line may carry a "> " blockquote marker; strip it along with any trailing "**".
    abstract_m = re.search(r"_Abstract_\s*\n\n?([\s\S]+?)(?=\s*\*{0,2}Keywords|\Z)", text)
    if abstract_m:
        raw = abstract_m.group(1).strip().rstrip("*").strip()
        raw = re.sub(r"^>\s*", "", raw)
        fields["abstract"] = " ".join(raw.split())

    return fields



def _build_raw_frontmatter_cell(fields: dict, original_cell: dict, nb_repo_path: str) -> dict:
    """Build a raw YAML front-matter cell from parsed fields.

    nb_repo_path: path of the source notebook relative to the repo root,
                  e.g. 'notebooks/model_demos/stommel.ipynb'. Used to build
                  the binder and download URLs injected as other-links.
    """
    UNQUOTED = {"last-modified", "today", "now"}

    binder_url = (
        f"https://mybinder.org/v2/gh/{GITHUB_REPO}/HEAD"
        f"?labpath={nb_repo_path}"
    )
    download_url = (
        f"https://raw.githubusercontent.com/{GITHUB_REPO}"
        f"/{GITHUB_BRANCH}/{nb_repo_path}"
    )

    lines = ["---\n"]
    for key in ("title", "author", "date"):
        if key in fields:
            val = fields[key]
            if val in UNQUOTED:
                lines.append(f"{key}: {val}\n")
            else:
                lines.append(f'{key}: "{val}"\n')
    if "abstract" in fields:
        lines.append(f"abstract: >\n  {fields['abstract']}\n")
    if "keywords" in fields:
        lines.append(f"keywords: {fields['keywords']}\n")
    lines.append(f'other-links:\n')
    lines.append(f'  - text: "Launch on Binder"\n')
    lines.append(f'    href: "{binder_url}"\n')
    lines.append(f'    icon: rocket-takeoff\n')
    lines.append(f'  - text: "Download notebook"\n')
    lines.append(f'    href: "{download_url}"\n')
    lines.append(f'    icon: download\n')
    lines.append("---")

    new_cell = copy.deepcopy(original_cell)
    new_cell["cell_type"] = "raw"
    new_cell["source"] = lines
    new_cell.pop("attachments", None)
    return new_cell


def _convert_notebook(nb: dict, nb_repo_path: str) -> tuple[dict, bool]:
    """Return (notebook, was_converted). Converts markdown metadata cell if present."""
    if not nb["cells"]:
        return nb, False
    first = nb["cells"][0]
    if first["cell_type"] != "markdown":
        return nb, False
    fields = _parse_markdown_frontmatter(first["source"])
    if not fields:
        return nb, False

    nb = copy.deepcopy(nb)
    nb["cells"][0] = _build_raw_frontmatter_cell(fields, first, nb_repo_path)
    return nb, True


def sync():
    DST.mkdir(parents=True, exist_ok=True)
    for nb_path in sorted(SRC.glob("*.ipynb")) + sorted(SRC.glob("*.qmd")):
        dst = DST / nb_path.name
        if not dst.exists() or nb_path.stat().st_mtime > dst.stat().st_mtime:
            if nb_path.suffix == ".ipynb":
                nb = json.loads(nb_path.read_text(encoding="utf-8"))
                nb_repo_path = nb_path.relative_to(REPO_ROOT).as_posix()
                nb, converted = _convert_notebook(nb, nb_repo_path)
                dst.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
                tag = " (front matter converted)" if converted else ""
                print(f"  synced {nb_path.relative_to(REPO_ROOT)} → docs/notebooks/{nb_path.name}{tag}")
            else:
                shutil.copy2(nb_path, dst)
                print(f"  synced {nb_path.relative_to(REPO_ROOT)} → docs/notebooks/{nb_path.name}")

if __name__ == "__main__":
    print("sync_notebooks: copying notebooks into docs/notebooks/")
    sync()
    print("sync_notebooks: done")
