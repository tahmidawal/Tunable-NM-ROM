"""Post-opening replay of the registered ST matched-FOM timing analysis.

Use --upstream to select the pinned plateau source and --raw-root for the
extracted sealed{dim}d{mesh}/output/result.json archives. --lane defaults
to this script's parent, with F1.json, configs/, receipts/ and results/.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from statistics import median


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def checked(path, expect):
    data = path.read_bytes()
    require(sha(data) == expect, f"SHA256 mismatch at {path}")
    return data


def samples(rows, name, cases, reps):
    """Milliseconds for verified same-output invocations in one block."""
    out = {}
    for row in rows:
        if row["kind"] + "|" + row["name"] == name:
            key = row["case"], row["rep"]
            require(key not in out and 0 <= row["diff"] <= 1e-12, "duplicate invocation or output mismatch")
            require(math.isfinite(row["seconds"]) and row["seconds"] > 0, "invalid time")
            out[key] = row["seconds"] * 1000
    require(set(out) == {(c, r) for c in cases for r in range(reps)}, "incomplete timing block")
    return out


def accuracy(stats):
    return {k: stats[k] for k in ("W", "T", "W_lo", "W_hi", "T_lo", "T_hi")}


def one_mesh(lane, root, frozen, plateau, dim, mesh):
    task = f"sealed{dim}d{mesh}"
    receipts = lane / "receipts" / task
    hashes = {line.split(None, 1)[1].strip(): line.split()[0]
              for line in (receipts / "OUTPUTS.sha256").read_text().splitlines()}
    result_sha, audit_sha = hashes["output/result.json"], hashes["output/witness-audit.json"]
    rep = json.loads(checked(root / task / "output/result.json", result_sha))
    audit = json.loads(checked(receipts / "witness-audit.json", audit_sha))
    cfg_name = f"configs/d{dim}m{mesh}.json"
    cfg_sha = frozen["files"][cfg_name]
    cfg = json.loads(checked(lane / cfg_name, cfg_sha))
    decision_raw = (lane / "results" / f"{task}-decision.json").read_bytes()
    decision = json.loads(decision_raw)
    require(rep["complete"] and cfg == rep["config"] and cfg["cases"] == rep["cases"], "unexpected panel")
    require(audit["passed"] and audit["result_sha256"] == decision["result_sha256"] == result_sha
            and str(audit["job_id"]) == str(decision["job_id"]) == str(rep["job_id"]), "inconsistent witnesses")
    rule = {2: "fib987", 3: "lat4096"}[dim]
    status = decision["kinds"]["ST"]["rules"][rule]["decision"]
    out = dict(job_id=rep["job_id"], gpu=rep["gpu"], quadrature_rule=rule, primary_ST=status,
               result_sha256=result_sha, audit_sha256=audit_sha, config_sha256=cfg_sha,
               decision_sha256=sha(decision_raw))
    bank = plateau.arm_stats(rep, "ST")[rule]
    selection = plateau.matched_fom(rep, rule, "ST")
    if status != "PASS" or selection is None:
        return dict(out, qualified=False, reasons=["plateau or FOM accuracy match unavailable"])
    name = selection["fom"]
    full = plateau.fom_stats(rep, "ST")[name]
    require(bank["valid"] and full["converged_all"] and bank["n"] == full["n"]
            and all(full[k+"_hi"] <= bank[k+"_lo"] for k in ("W", "T")), "invalid ST match")
    phase = rep["timing"]
    lattice, fom = "rom_lat|"+rule, "fom|"+name
    cases = phase["cases"]
    require(cases == cfg["cases"][:6] and cfg["timing"]["reps"] == 3
            and cfg["timing"]["confirm_reps"] == 5, "unexpected timing inventory")
    subjects = {}
    confirm = {}
    for label in (lattice, fom):
        m = median(samples(phase["invocations"], label, cases, 3).values())
        confirm[label] = samples(phase["confirm_invocations"], label, cases, 5)
        c = median(confirm[label].values())
        require(math.isclose(m, phase["summary"][label]["median_ms"], rel_tol=1e-12)
                and math.isclose(c, phase["confirm_summary"][label]["median_ms"], rel_tol=1e-12), "timing mismatch")
        subjects[label] = dict(main_ms=m, confirm_ms=c, median_fold_change=max(m/c, c/m))
    per_case, ratios, sensitivity = [], [], []
    for case in cases:
        rom_ms = median(confirm[lattice][case, r] for r in range(5))
        fom_ms = median(confirm[fom][case, r] for r in range(5))
        row = next(r for r in rep["fom_rows"] if r["fom"] == name and r["case"] == case)
        prep = row.get("prepare_seconds", row.get("evidence", {}).get("prepare_seconds"))
        if prep is not None:
            require(math.isfinite(prep) and prep >= 0, "invalid input preparation time")
            sensitivity.extend(confirm[fom][case, r] + prep*1000 for r in range(5))
        ratios.append(rom_ms/fom_ms)
        per_case.append(dict(case=case, rom_ms=rom_ms, fom_ms=fom_ms, rom_over_fom=rom_ms/fom_ms,
                             fom_input_preparation_ms=None if prep is None else 1000*prep))
    require(not sensitivity or len(sensitivity) == 30, "incomplete preparation sensitivity")
    gate = rep["gates"]["G_time"]
    drift = max(v["block_drift"] for k in ("summary", "confirm_summary") for v in phase[k].values())
    require(math.isclose(drift, gate["worst_block_drift"], rel_tol=1e-12)
            and gate["drift_pass"] is (drift <= 1.10), "inconsistent timing gate")
    reasons = []
    if not gate["outputs_pass"]:
        reasons.append("panel timed-output agreement")
    if not gate["drift_pass"]:
        reasons.append("panel repetition-block drift")
    for label, val in subjects.items():
        if val["median_fold_change"] > 1.10:
            reasons.append(f"{label} main/confirmation drift")
    rom_ms, fom_ms = subjects[lattice]["confirm_ms"], subjects[fom]["confirm_ms"]
    return dict(out, matched_fom=name, eligible_ST=bank["n"], all_fom_cases_converged=full["n_cases"],
                ST=dict(rom=accuracy(bank), fom=accuracy(full)), timing_case_ids=cases,
                main_invocations_per_subject=18, confirm_invocations_per_subject=30,
                qualified=not reasons, reasons=reasons, gates=gate, subject_timings=subjects,
                confirm_rom_over_fom=rom_ms/fom_ms, confirm_fom_over_rom=fom_ms/rom_ms, per_case_confirm_ms=per_case,
                paired_rom_over_fom=dict(minimum=min(ratios), median=median(ratios), maximum=max(ratios)),
                fom_ms_with_input_preparation_sensitivity=median(sensitivity) if sensitivity else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    a = parser.parse_args()
    review = json.loads((a.lane / "F1-REVIEW.json").read_bytes())
    require(review["decision"] == "ACCEPT", "unreviewed freeze")
    frozen = json.loads(checked(a.lane / "F1.json", review["freeze_sha256"]))
    upstream = json.loads(checked(a.lane / "upstream.json", frozen["files"]["upstream.json"]))
    path = a.upstream / "plateau.py"
    checked(path, upstream["files"]["plateau.py"])
    module = importlib.util.spec_from_file_location("history_plateau", path)
    plateau = importlib.util.module_from_spec(module)
    module.loader.exec_module(plateau)
    output = dict(freeze_sha256=review["freeze_sha256"], upstream_commit=upstream["commit"],
                  matching_source_sha256=upstream["files"]["plateau.py"], generator_sha256=sha(Path(__file__).read_bytes()),
                  selection="The pinned matched_fom implementation chooses the fastest main-panel FOM among all-case-converged "
                            "subjects meeting the frozen lattice's eligible ST maximum AND median with adverse uncertainty bounds. "
                            "The same allocation supplies five confirmation repetitions on each of six fixed cases for this pair.",
                  scope="Both solvers return six fields on the scoring lattice. The timer begins with a prepared initial field for "
                        "the bank and prepared initial sine coefficients for the matched spectral FOM. The bank projection, "
                        "solve and lattice decode are included. Setup, compilation and preparing both inputs are excluded.",
                  interpretation="A confirmation ROM/FOM ratio above one favors the FOM. Qualified requires timed-output agreement, "
                                 "global repetition-block drift and both selected members' main/confirmation agreement. The separate "
                                 "sensitivity adds each case's singly measured, synchronized FOM preparation to each of its five timings. "
                                 "It omits the ROM's input preparation and is not an end-to-end comparison.",
                  panels={})
    for dim, meshes in ((2,(256,512,1024,2048,4096)), (3,(65,129,257))):
        for mesh in meshes:
            output["panels"][f"sealed{dim}d{mesh}"] = one_mesh(a.lane,a.raw_root,frozen,plateau,dim,mesh)
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
