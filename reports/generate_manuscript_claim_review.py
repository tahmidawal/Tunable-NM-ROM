"""Render the claim review against its exact historical manuscript snapshot."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "reports/2026-09-20-manuscript-claim-audit.json"


def main():
    audit = json.loads(PATH.read_text())
    source = subprocess.check_output(
        ["git", "show", audit["source_commit"] + ":paper/main.tex"], cwd=ROOT)
    assert hashlib.sha256(source).hexdigest() == audit["source_sha256"]
    source = source.decode()
    lines = ["# Manuscript claim scope review", "",
        "This is a source review of the current manuscript, with proposed wording corrections "
        "awaiting the manuscript owner. It changes no measured result.", "",
        "| Location | Claim issue | Required scope |", "| --- | --- | --- |"]
    for row in audit["findings"]:
        number = source[:source.index(row["source_excerpt"])].count("\n") + 1
        assert number == row["line"]
        lines.append(f"| `main.tex:{number}` | {row['issue']} | {row['proposed_direction']} |")
    lines += ["", "The accompanying JSON pins the reviewed source and exact excerpts. "
        "Final disposition must identify the repaired source commit, retain historical wording "
        "in Git, and distinguish wording changes from any new empirical result.", "",
        "## Glossary", "",
        "- **Frozen decoder/operator:** a neural model whose trained weights stay unchanged during evaluation.",
        "- **Trial space:** the set of solution fields the reduced model can represent.",
        "- **Residual test space:** the smooth functions used to measure the PDE residual.",
        "- **Projection / affine map:** restriction to a chosen lower-dimensional representation / a linear map with a constant offset.",
        "- **Reduction error / discretization error:** discrepancy from the chosen discrete equations / discrepancy caused by approximating the continuous equations on a grid.",
        "- **Disposition:** the recorded response to each review finding.", ""]
    PATH.with_suffix(".md").write_text("\n".join(lines))
    print(f"Rendered {len(audit['findings'])} pinned manuscript findings.")


if __name__ == "__main__":
    main()
