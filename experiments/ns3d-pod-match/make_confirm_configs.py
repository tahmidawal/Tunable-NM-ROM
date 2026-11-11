"""Configs for confirm_mesh.py (PROTOCOL.md): development and test cohorts at 32/64/96 and
the exploratory 128^3 development run. Test configs are the parent lane's test configs
with this lane's frozen bank; nothing else about the cohort or the references changes."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
dev_src = {32: "dev32b", 64: "dev64", 96: "dev96"}
d64 = json.loads((HERE / "configs/dev64.json").read_text())
interp = dict(source_n=64, dir=d64["pod_frozen_dir"], sha256=d64["pod_frozen_sha256"],
              spans=[8, 16, 32, 64])
coarse = dict(meshes=[12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80],
              cnab2_steps=[5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 200],
              rk4_steps=[5, 10, 15, 20, 30, 40], time_below=0.05, include_same_mesh=True)
smoke = dict(eval_cases=2, reference_errors=None, timing_repetitions=3, cnab2_steps=[200, 40],
             neighbour_gate_ratio=10.0,
             coarse=dict(meshes=[16, 24], cnab2_steps=[20, 40], rk4_steps=[10], time_below=0.05,
                         include_same_mesh=True))
for n, name in dev_src.items():
    dev = json.loads((HERE / f"configs/{name}.json").read_text())
    dev.update(name=f"confirm_dev{n}", interp_pod=interp, coarse=coarse, smoke=smoke)
    (HERE / f"configs/confirm_dev{n}.json").write_text(json.dumps(dev, indent=1) + "\n")
    test = json.loads((HERE / f"../ns3d-coordnet/configs/test{n}.json").read_text())
    for key in ("coordnet_bank_file", "coordnet_bank_sha256", "one_bank_orthonormality_gate",
                "bank_prefix", "span_ladder"):
        test[key] = dev[key]
    for key in ("k", "ladder_dt", "ladder_iters", "damping", "ic_iters", "modes", "train_seed",
                "pod_frozen_dir", "pod_frozen_sha256", "dt_truth", "horizon"):
        assert test[key] == dev[key], key
    assert test["eval_seed"] == 202609221 and test["eval_cases"] == 32
    test.update(name=f"confirm_test{n}", interp_pod=interp, coarse=coarse)
    (HERE / f"configs/confirm_test{n}.json").write_text(json.dumps(test, indent=1) + "\n")
big = json.loads((HERE / "configs/confirm_dev96.json").read_text())
for key in ("reference_errors", "reference_source", "smoke"):
    big.pop(key, None)
big.update(name="confirm_dev128", n=128, pod_frozen_dir=None, pod_frozen_sha256={},
           save_fields=False, controls=False, cnab2_steps=[200, 100, 80, 70, 60, 50, 40],
           coarse=dict(coarse, meshes=[24, 32, 40, 48, 56, 64, 72, 80, 96]))
(HERE / "configs/confirm_dev128.json").write_text(json.dumps(big, indent=1) + "\n")
fz = HERE / "frozen/frozen_settings.json"
f = json.loads(fz.read_text())
f["heads"]["128"] = dict(f["heads"]["96"], trained_n=96,
                        note="96^3 head used zero-shot at 128^3 (exploratory)")
fz.write_text(json.dumps(f, indent=1) + "\n")
