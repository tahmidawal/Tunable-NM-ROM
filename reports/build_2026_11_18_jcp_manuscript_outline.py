"""Render the manuscript outline from a prose template and immutable measured data.

This CPU-only builder runs no scientific model, search, or selection experiment.
"""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "reports/2026-11-18-jcp-manuscript-outline-assets"
CAMPAIGN = ROOT / "reports/2026-11-18-jcp-controlled-experiments-assets"
AUDIT = ROOT / "reports/2026-11-18-jcp-results-audit-assets"
REPORT = ROOT / "reports/2026-11-18-jcp-manuscript-outline.md"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = CAMPAIGN / "measured-values.json"
    source_bytes = source.read_bytes()
    frozen_source = ASSETS / "campaign-measured-values.json"
    if frozen_source.exists():
        assert frozen_source.read_bytes() == source_bytes, "Registered source changed; review before revising this outline."
    else:
        frozen_source.write_bytes(source_bytes)
    data = json.loads(source_bytes)
    head = next(v for v in data["head-p03b"] if v["rows"] == 584 and v["dt"] == .01)
    widths = data["weak-width-final"]
    narrow = min(widths, key=lambda v: (v["R"], v["M"]))
    wide = max(widths, key=lambda v: (v["R"], v["M"]))
    costs = data["cost-p04"]
    candidates = [v for v in costs if v["target"].startswith("learned_head") and v["group"] == "pod"]
    cheapest = min(candidates, key=lambda v: v["target_median_ms"])
    coarse = next(v for v in costs if v["target"] == cheapest["target"] and v["group"] == "any_mesh_classical")
    assert cheapest["qualified"] and coarse["qualified"]
    assert all(v["target_faster_case_count"] == 0 for v in costs if v["target"].startswith("learned_head") and v["group"] == "any_mesh_classical")
    rows = [
        "| Evidence | Generated numerical anchor | Required interpretation |",
        "|---|---|---|",
        f"| Matched head/span, M={head['rows']}, $\\Delta t={head['dt']:g}$ | Median maximum evolved error: {head['head_median_percent']:.4f}% / {head['span_median_percent']:.4f}%; median paired span/head ratio {head['paired_ratio_median']:.2f}; head better in {head['head_lower_count']}/{len(head['cases'])} cases | Original-family learned bank; common initial field; initial-field-normalized norm; completed development evidence |",
        f"| Frozen Burgers span and tests | Worst initialization-inclusive error: {100*narrow['maximum_error_worst']:.4f}% at R′={narrow['R']}, M={narrow['M']}, to {100*wide['maximum_error_worst']:.4f}% at R′={wide['R']}, M={wide['M']}; {wide['qualified_count']}/{wide['case_count']} cases qualify | Dense discrete-upwind dynamics at fixed time step; width and test count both change; not head or sparse-cost evidence |",
        f"| Cheapest admitted head versus eligible alternatives | Complete-query medians: head {cheapest['target_median_ms']:.2f} ms; POD {cheapest['comparator_median_ms']:.2f} ms; coarse classical {coarse['comparator_median_ms']:.2f} ms | Same allocation; cohort median-and-worst accuracy eligibility; near POD tie and no coarse-classical advantage |",
    ]
    template = ASSETS / "outline-template.md"
    report = template.read_text()
    for key, val in {"ROOT":str(ROOT), "ASSETS":str(ASSETS), "AUDIT":str(AUDIT), "CAMPAIGN":str(CAMPAIGN), "ANCHOR_TABLE":"\n".join(rows)}.items():
        report = report.replace("{{"+key+"}}", val)
    assert not re.search(r"\{\{[A-Z_]+\}\}", report)
    ASSETS.mkdir(parents=True, exist_ok=True)
    manifest = {
        "purpose":"Generated numerical anchors for a manuscript outline; no new experiment",
        "sources":[{"path":str(source), "frozen":str(frozen_source), "sha256":sha(source)},
                   {"path":str(CAMPAIGN / "source-manifest.json"), "role":"upstream raw-evidence provenance", "sha256":sha(CAMPAIGN / "source-manifest.json")}],
        "template":{"path":str(template), "sha256":sha(template)},
        "derived":{"head":head, "width_baseline":narrow, "width_largest_registered":wide, "lowest_median_cost_head_pod_comparison":cheapest, "matching_coarse_classical_comparison":coarse},
    }
    manifest_path = ASSETS / "evidence-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    REPORT.write_text(report)
    checked = []
    for raw in re.findall(r"\]\(([^)]+)\)", report):
        if raw.startswith(("https://", "http://", "#")):
            continue
        target = re.sub(r":\d+$", "", raw.split("#",1)[0])
        path = Path(target)
        if not path.is_absolute():
            path = REPORT.parent / path
        assert path.exists(), path
        checked.append(str(path))
    verification = {"status":"passed", "local_links_checked":len(checked), "report_sha256":sha(REPORT), "manifest_sha256":sha(manifest_path), "rendered_word_count":len(re.findall(r"\b[\w′–-]+\b", re.sub(r"\]\([^)]*\)", "]", report)))}
    (ASSETS / "verification.json").write_text(json.dumps(verification, indent=2)+"\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
