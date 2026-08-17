"""Inline every figure of report_src.md as a data URI -> Coordinate-ROM-Findings.md.

report_src.md is the editable source (relative image links, readable in any
editor). The built file is what gets published as the artifact, where relative
paths cannot resolve.
"""
from __future__ import annotations

import base64
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "report_src.md")
OUT = os.path.join(HERE, "Coordinate-ROM-Findings.md")
SEARCH = [HERE, "/home/tahmid/Dev/pod-ae-nmrom"]


def resolve(rel: str) -> str:
    for root in SEARCH:
        p = os.path.join(root, rel)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(rel)


def inline(match: re.Match) -> str:
    alt, rel = match.group(1), match.group(2)
    if rel.startswith("data:"):
        return match.group(0)
    path = resolve(rel)
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    return f"![{alt}](data:image/png;base64,{b64})"


text = open(SRC).read()
n = len(re.findall(r"!\[([^\]]*)\]\(([^)]+)\)", text))
text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", inline, text)
open(OUT, "w").write(text)
print(f"inlined {n} figures -> {OUT} ({os.path.getsize(OUT)/1e6:.2f} MB)")
