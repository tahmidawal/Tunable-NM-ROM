"""Render README.md from README.tmpl.md by substituting facts derived from the JSONs.

Every `{{name}}` in the template is replaced by `wsf_facts.build()[name]`.  An unknown
placeholder is a hard error, so the README cannot quote a number that the data does
not support, and a stale number cannot survive a re-run.  This exists because the
previous round's audit found zero errors in generated tables and 19 in hand-written
prose.

Usage: python wsf_render_readme.py [runs_dir]
"""
from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wsf_facts  # noqa: E402

TMPL = os.path.join(HERE, "README.tmpl.md")
OUT = os.path.join(HERE, "README.md")


def main():
    f = wsf_facts.build(sys.argv[1] if len(sys.argv) > 1 else None)
    src = open(TMPL).read()
    missing = sorted({m for m in re.findall(r"\{\{([A-Za-z0-9_]+)\}\}", src)
                      if m not in f})
    if missing:
        raise SystemExit("README.tmpl.md references facts that do not exist in the "
                         "data: " + ", ".join(missing))
    out = re.sub(r"\{\{([A-Za-z0-9_]+)\}\}", lambda m: f[m.group(1)], src)
    used = sorted(set(re.findall(r"\{\{([A-Za-z0-9_]+)\}\}", src)))
    with open(OUT, "w") as fh:
        fh.write(out)
    print(f"wrote {OUT} ({len(used)} facts substituted, {len(f)} available)")


if __name__ == "__main__":
    main()
