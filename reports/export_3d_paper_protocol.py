"""Export experimental setup and recorded operator training for selected panels.

Read-only with respect to experiment trees. Requested training budgets never
stand in for completed updates, and reused records never imply fresh training.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT = REPORTS / "2026-09-20-3d-paper-panels"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def setup(entry, data, normalized):
    cfg = data["config"]
    common = dict(pde=entry["pde"], attempt=entry["attempt"],
                  timed_cases=sorted({r["cases"] for r in normalized["rows"]}),
                  repetitions=cfg["repetitions"], f64=data["x64"],
                  precision=data.get("precision", data.get("matmul_precision")))
    if entry["adapter"] == "burgers":
        return dict(common, training_cases=cfg["train_trajectories"],
                    training_fields_per_case=len(cfg["train_steps"]),
                    outputs_per_query=cfg["steps"]+1, components=1,
                    bank_rank=cfg["training"]["rank"],
                    latent_dimensions=[cfg["training"]["latent_dimension"]],
                    weak_tests=data["actual_test_modes"], requested_weak_tests=cfg["test_modes"], q=cfg["q_ladder"],
                    training_mesh=cfg["nodes"], mesh_convention="nodes per axis")
    if entry["adapter"] == "ns_trajectory":
        rows = read(ROOT / entry["invocations"])
        outputs = len(rows[0]["same_grid_errors"])
        assert all(len(r["same_grid_errors"]) == outputs for r in rows)
        return dict(common, training_cases=cfg["train_cases"],
                    training_fields_per_case=outputs, outputs_per_query=outputs,
                    components=3, bank_rank=data["selected_rank"],
                    latent_dimensions=[cfg["k"]], weak_tests=cfg["test_modes"], requested_weak_tests=cfg["test_modes"],
                    q=cfg["q_values"], training_mesh=cfg["n"],
                    mesh_convention="periodic points per axis")
    assert entry["adapter"] in {"heat", "poisson"}
    outputs = len(cfg["times"]) if entry["adapter"] == "heat" else 1
    return dict(common, training_cases=cfg["train_count"],
                training_fields_per_case=outputs, outputs_per_query=outputs,
                components=1, bank_rank=cfg["bank_rank"],
                latent_dimensions=sorted({head["k"] for head in data["heads"]}),
                weak_tests=cfg["weak_tests"], requested_weak_tests=cfg["weak_tests"],
                q=cfg["q_ladder"], training_mesh=cfg["train_intervals"],
                mesh_convention="intervals per axis")


def main():
    paths = [REPORTS / "2026-09-20-3d-paper-panel-selection.json",
             REPORTS / "2026-09-20-3d-paper-results.json", Path(__file__)]
    selection, results = map(read, paths[:2])
    provenance = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    setups, operators = [], []
    for panel in selection["panels"]:
        entry = next(r for r in results["runs"]
                     if (r["pde"], r["attempt"]) == (panel["pde"], panel["attempt"]))
        path = ROOT / entry["result"]
        data = read(path)
        assert data["complete"] and entry["audit"]["passed"]
        provenance[entry["result"]] = sha(path)
        row = setup(entry, data, entry)
        row["source"] = entry["source"]
        row["qualification"] = panel["qualification"]
        row["raw_data_contract"] = data.get("data_contract")
        row["raw_output_contract"] = data.get("output_contract", data.get("operators", {}).get("output_contract")
                                               if isinstance(data.get("operators"), dict) else None)
        setups.append(row)
        infos = data["operators"]["models"] if isinstance(data["operators"], dict) else data["operators"]
        for info in infos:
            assert info["parameter_dtype"] == "float64" and info["fft_dtype"] == "complex128"
            cfg = info["config"]
            assert 0 < info["best_step"] <= info["steps_completed"] <= cfg["steps"]
            operators.append(dict(pde=entry["pde"], attempt=entry["attempt"],
                name=info.get("name", info["spec"]["kind"]),
                parameters=info["parameter_count"], requested_steps=cfg["steps"],
                completed_steps=info["steps_completed"], selected_step=info["best_step"],
                training_seed=cfg["seed"], batch_size=cfg["batch_size"],
                initial_learning_rate=cfg["learning_rate"], exit_reason=info["exit_reason"],
                spec=info["spec"], raw_record=info))
    report = ["# Experimental setup and operator training for the 3D panels", "",
        "These generated tables describe the selected audited development panels. "
        "They are provisional; final testing and remaining tuning are tracked in the canonical lab log.", "",
        "The original run configuration and saved training record supply every numeric cell. "
        "Training records may be reused from earlier attempts; this table does not claim that training "
        "occurred inside the query-timing allocation or that different architectures received equal compute.", "",
        "| PDE | Training cases | Fields / training case | Outputs / query | Components | Training grid | Bank R | Heads K | Weak tests M | Correction ranks q |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | --- |"]
    for r in setups:
        report.append(f"| {r['pde']} | {r['training_cases']} | {r['training_fields_per_case']} | "
                      f"{r['outputs_per_query']} | {r['components']} | {r['training_mesh']} {r['mesh_convention']} | "
                      f"{r['bank_rank']} | {', '.join(map(str,r['latent_dimensions']))} | {r['weak_tests']} | "
                      f"{', '.join(map(str,r['q']))} |")
    report += ["", "A field means a complete spatial state; a vector state includes all velocity components. "
        "Time-zero fields are included in the counts when requested. Operators may pass through the supplied "
        "initial field and interpolate trained output times, as specified by the corresponding panel. "
        "The counts describe training membership, not statistically independent state samples. "
        "Weak-test counts use the actual basis size; Burgers completes the degenerate Laplacian "
        "eigenvalue shell at the cutoff to preserve coordinate symmetry. The JSON also retains the requested count.", "",
        "| PDE | Operator | Parameters | Requested updates | Completed updates | Selected update | Batch | Initial learning rate | Training seed | Exit |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in operators:
        report.append(f"| {r['pde']} | `{r['name']}` | {r['parameters']} | {r['requested_steps']} | "
                      f"{r['completed_steps']} | {r['selected_step']} | {r['batch_size']} | "
                      f"{r['initial_learning_rate']:.6g} | {r['training_seed']} | {r['exit_reason']} |")
    report += ["", "Checkpoint selection uses the recorded development criterion. "
        "Reaching an update or elapsed-time budget is not evidence that training converged. "
        "The adjacent JSON retains architecture settings, normalization, original training metadata and source hashes.", "",
        "## Glossary", "",
        "- **PDE / grid:** differential equation / number of spatial nodes or intervals along each axis, using the stated convention.",
        "- **Training case / field / output:** one sampled physical input / one spatial state / one requested state returned by a query.",
        "- **Components:** scalar channels in a physical field; velocity is vector valued.",
        "- **Bank R / head K / weak tests M / correction q:** learned spatial basis size / nonlinear code size / number of smooth residual tests / added linear correction directions.",
        "- **Operator / parameters:** a trained solution-map model / its recorded trainable parameter count.",
        "- **Requested / completed / selected updates:** configured optimizer budget / actual updates executed / update supplying the retained validation-selected checkpoint.",
        "- **Batch / initial learning rate / seed:** examples per optimizer update / starting step-size setting before its recorded schedule / reproducible random initializer.",
        "- **Exit:** recorded reason training stopped; an update limit or wall-time limit is not a convergence certificate.",
        "- **Development / provisional:** inputs available during selection / evidence still awaiting final confirmation.", ""]
    (OUT / "protocol.md").write_text("\n".join(report))
    (OUT / "protocol.json").write_text(json.dumps(dict(status=selection["status"],
        provenance=provenance, setups=setups, operators=operators), indent=2)+"\n")
    with (OUT / "operator-training.csv").open("w", newline="") as stream:
        keys = [k for k in operators[0] if k not in {"spec", "raw_record"}]
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader(); writer.writerows(operators)
    print(f"Exported {len(setups)} experimental setups and {len(operators)} recorded operator recipes.")


if __name__ == "__main__":
    main()
