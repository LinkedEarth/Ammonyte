"""Inject figure references into quartodoc-generated reference pages.

Scans every ``docs/reference/*.qmd`` file for Markdown fenced code blocks
that contain a ``plt.savefig(...)`` call pointing to ``docs/reference/figures/``.
Where found, inserts a Markdown image reference immediately after the code
block — but only if one is not already present (idempotent).

Run this after ``quartodoc build`` and before ``quarto preview`` / ``quarto render``.
It does not require the figure files to exist; missing figures produce a
broken-image placeholder in Quarto but do not prevent the build from succeeding.

Usage
-----
Inject into all reference pages::

    python scripts/inject_doc_figures.py

Inject into a single file::

    python scripts/inject_doc_figures.py docs/reference/Lorenz63.qmd

Dry-run (print changes without writing)::

    python scripts/inject_doc_figures.py --dry-run
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QMD_DIR = ROOT / 'api'

# Match the savefig path inside a code block, capturing the bare filename.
# Handles both single- and double-quoted paths.
SAVEFIG_RE = re.compile(
    r"""plt\.savefig\(\s*['"]docs/reference/figures/([^'"]+\.png)['"]"""
)


def _make_alt(figname: str) -> str:
    """Derive a readable alt-text string from a figure filename."""
    stem = figname.replace('_example.png', '')
    # CamelCase → words with spaces, e.g. Lorenz63 → Lorenz63
    return f'{stem} example output'


def inject_into_content(content: str) -> tuple[str, list[str]]:
    """Process the text of one .qmd file.

    Detects Markdown fenced code blocks (`` ```python `` or plain `` ``` ``).
    If a block contains a ``plt.savefig(...)`` call pointing to
    ``api/figures/``, a Markdown image reference is inserted
    immediately after the closing fence.

    Returns the (possibly modified) content and a list of injected filenames.
    """
    lines = content.split('\n')
    result: list[str] = []
    injected: list[str] = []
    in_block = False
    block_lines: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # Opening fence: line starts with ``` (any language tag)
        if not in_block and line.startswith('```'):
            in_block = True
            block_lines = [line]
            result.append(line)
            i += 1
            continue

        if in_block:
            block_lines.append(line)
            result.append(line)

            # Closing fence: bare ``` with optional trailing whitespace,
            # and at least one content line seen (len > 1 guards against the
            # opening fence itself matching)
            if line.rstrip() == '```' and len(block_lines) > 1:
                in_block = False
                block_text = '\n'.join(block_lines)

                m = SAVEFIG_RE.search(block_text)
                if m:
                    figname = m.group(1)

                    # Idempotency: skip if the next non-blank line is already
                    # an image reference.
                    j = i + 1
                    while j < len(lines) and lines[j].strip() == '':
                        j += 1
                    next_nonempty = lines[j].strip() if j < len(lines) else ''

                    if not next_nonempty.startswith('!['):
                        alt = _make_alt(figname)
                        result.append('')
                        result.append(f'![{alt}](../figures/{figname})')
                        result.append('')
                        injected.append(figname)

                block_lines = []

            i += 1
            continue

        result.append(line)
        i += 1

    return '\n'.join(result), injected


def process_file(path: Path, dry_run: bool = False) -> int:
    """Inject figure references into *path*.  Returns number of injections."""
    original = path.read_text(encoding='utf-8')
    modified, injected = inject_into_content(original)

    if not injected:
        return 0

    print(f'  {path.relative_to(ROOT)}')
    for fig in injected:
        print(f'    + {fig}')

    if not dry_run:
        path.write_text(modified, encoding='utf-8')

    return len(injected)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'files', nargs='*', metavar='FILE',
        help='Specific .qmd files to process (default: all api/*.qmd)',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print what would be injected without modifying any files.',
    )
    args = parser.parse_args()

    if args.files:
        targets = [Path(f) for f in args.files]
    else:
        targets = sorted(QMD_DIR.glob('*.qmd'))

    if not targets:
        print('No .qmd files found.')
        sys.exit(0)

    if args.dry_run:
        print('Dry run — no files will be modified.\n')

    total = 0
    for path in targets:
        if not path.exists():
            print(f'  WARNING: {path} not found, skipping.')
            continue
        total += process_file(path, dry_run=args.dry_run)

    if total == 0:
        print('Nothing to inject (all figures already referenced or no savefig calls found).')
    else:
        action = 'Would inject' if args.dry_run else 'Injected'
        print(f'\n{action} {total} figure reference(s).')


if __name__ == '__main__':
    main()
