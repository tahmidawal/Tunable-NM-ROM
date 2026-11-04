"""Write the bank-job configs of ns3d-pod-match (arms a and b) from one shared base.

The shared base is the A3 production config `ns3d-coordnet/configs/bank_r64a.json`
(same seeds, data, compression, floor meshes, parent POD references). Run:
  /home/tahmid/Dev/.venv/bin/python experiments/ns3d-pod-match/make_configs.py
"""
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
base = json.loads((HERE.parent / "ns3d-coordnet/configs/bank_r64a.json").read_text())
for key in ("architectures", "name", "smoke"):
    base.pop(key)

ARCH_A = dict(name="A", width=512, depth=4, kmax=4)           # the pilot's architecture
common = dict(
    base,
    rank=64,
    architecture=ARCH_A,
    prefixes=[8, 16, 32, 48],
    derivative_meshes=[32, 64],
    checkpoint_seconds=900,
    adam=dict(train_seconds=10800, lr=1e-3, final_lr_ratio=1e-3, warmup=1000),
    lbfgs=dict(train_seconds=6000, memory=20, max_linesearch_steps=30,
               iterations_per_call=25, stall_rtol=1e-10, stall_atol=1e-15, stall_calls=8),
    smoke=dict(bank_cases=6, data_meshes=[8, 16], dt_truth=0.005, horizon=0.05, train_grid=12,
               compress_K=16, rank=4, log_every=10, dev_cases=2, floor_meshes=[8, 16],
               prefixes=[2], parent_pod_cases=4, gram_block=8, derivative_meshes=[8],
               train_seed=7, dev_seed=8, offgrid_points=256, parent_pod_floor_reference={},
               architecture=dict(name="s", width=16, depth=2, kmax=2),
               adam=dict(train_seconds=6, lr=1e-3, final_lr_ratio=1e-3, warmup=5),
               lbfgs=dict(train_seconds=6, memory=5, max_linesearch_steps=10,
                          iterations_per_call=5, stall_rtol=1e-10, stall_atol=1e-15, stall_calls=8)),
)

arm_a = copy.deepcopy(common)
arm_a.update(name="bank_a", objective="vp_band")
arm_a["architecture"].update(band_kcut=10, band_weight=1.0)
arm_a["smoke"]["architecture"].update(band_kcut=3, band_weight=1.0)

arm_b = copy.deepcopy(common)
arm_b.update(name="bank_b", objective="pod_regress", offgrid_points=8192, offgrid_seed=20261103)

(HERE / "configs").mkdir(exist_ok=True)
for cfg in (arm_a, arm_b):
    (HERE / "configs" / f"{cfg['name']}.json").write_text(json.dumps(cfg, indent=1) + "\n")
    print("wrote", cfg["name"])

# ---------------------------------------------------------------- DESIGN §A2 ----
# warm start from bank_a's bank of record; objective = arm (a) + mu * ||B^T B - I||_F^2 at 64^3
# (the G3 statistic itself, computed from the bank and grids only); L-BFGS only.
import hashlib  # noqa: E402

init = "experiments/ns3d-pod-match/runs/bank_a/output/bank_final.npz"
init_sha = hashlib.sha256((HERE.parents[1] / init).read_bytes()).hexdigest()
for tag, mu in (("mu1", 1.0), ("mu001", 1e-2)):
    c = copy.deepcopy(arm_a)
    c.update(name=f"bank_a2_{tag}", objective="vp_band_g3", init_from=init, init_sha256=init_sha)
    c["architecture"].update(g3_weight=mu, g3_mesh=64, g3_chunk=32768)
    c["adam"] = dict(c["adam"], train_seconds=0)
    c["lbfgs"] = dict(c["lbfgs"], train_seconds=21600)
    c["smoke"]["architecture"].update(g3_weight=mu, g3_mesh=16, g3_chunk=512)
    c["smoke"]["init_from"] = None
    (HERE / "configs" / f"{c['name']}.json").write_text(json.dumps(c, indent=1) + "\n")
    print("wrote", c["name"], init_sha)
