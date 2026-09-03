"""Regenerate every number quoted in the book.

Usage:  make verify        (or)  /tmp/mlenv/bin/python verify/run_all.py

Each chNN_*.py script prints the values used in that chapter's worked
examples, tables and exercise answers, in the form they appear in the text.
A chapter is not finished until its printed numbers match its script.
"""

import pathlib
import runpy
import sys

HERE = pathlib.Path(__file__).parent

def main():
    scripts = sorted(p for p in HERE.glob("ch*_*.py"))
    if not scripts:
        print("no verification scripts found", file=sys.stderr)
        return 1
    for path in scripts:
        print("=" * 70)
        print(f"== {path.name}")
        print("=" * 70)
        runpy.run_path(str(path), run_name="__main__")
        print()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
