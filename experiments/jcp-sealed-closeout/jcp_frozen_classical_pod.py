"""Native-mesh classical controls for the frozen prospective Burgers evaluation.

Run in a process separate from the bank drivers (several historical modules use
the same top-level names). Pass the staged experiments root containing sources
from the historical imp-multimesh and b3d-pod lanes and their hr dependencies.
prepare() does not read any cohort. The caller supplies its physical initial
field and viscosity; the provided descriptor helpers reproduce the original ICs.
"""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


HASH_2D = {
    "Phi.npy": "88fd4022184f94a3d755c8d470c65e727e2ac225f3e88d0069cd55b90c0b5bfb",
    "rules.npz": "626b63b0ec9f75f30852c73858dce1e46568631ac68d383c1a5af605af9b4043",
    "trust.json": "7879f2f1b27617827de3dab375ef110885adae2cf50c63c494c7516f8d655cc1",
}
HASH_3D = "7dc39cf9fbb5ea06d77001e4f8ccd69113dcdc8f22838f3cc12d0159b3431b2c"


def verify(path, expected):
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"Frozen POD input changed: {path.name}")


def prepare(dim, experiments, asset):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    if jax.default_backend() != "gpu" or os.environ["JAX_DEFAULT_MATMUL_PRECISION"] != "highest":
        raise RuntimeError("POD control requires GPU and highest precision.")
    root = Path(experiments).resolve()
    directory = {2: "imp-multimesh", 3: "b3d-pod"}[dim]
    if "rom" in sys.modules or "common" in sys.modules or "interface" in sys.modules:
        raise RuntimeError("Launch the classical control in its own fresh process.")
    sys.path.insert(0, str(root / directory))
    asset = Path(asset).resolve()
    if dim == 2:
        module = importlib.import_module("mmcore")
        for name, expected in HASH_2D.items():
            verify(asset / name, expected)
        basis = np.load(asset / "Phi.npy", mmap_mode="r", allow_pickle=False)
        phi = jnp.asarray(np.ascontiguousarray(basis[:, :256]))
        assert phi.shape == (255 ** 2, 256) and phi.dtype == jnp.float64
        radius = json.loads((asset / "trust.json").read_text())["256"]
        cell = module.Cell(256, phi, {256: radius}, log=lambda *_: None)
        with np.load(asset / "rules.npz", allow_pickle=False) as rules:
            data, _ = cell.ecsw_data(256, rules["n|256|k8"], rules["w|256|k8"])
        query = module.RM.make_query("ecsw", 2, 256, "common", trust=radius)
        sch = module.RM.schedule(2, "CNR", .005, "prod")
        name = "CPU-256_r256_k8_CNR"
        lattice, pipeline = module.H, module.PL
        certifier=module.RM.make_certifier("ecsw",2,256,module.H.NMAX[2])
        rom=module.RM
    else:
        module = importlib.import_module("interface")
        assert Path(module.__file__).resolve() == root / directory / "interface.py"
        # setup verifies all bytes of the stored training basis.
        assert module.PHI_SHA == HASH_3D
        _, _, _, _, data, _, meta = module.setup(asset)
        radius = meta["trust_radius"]
        assert radius == 3.097916703945634
        query, _, _, _, certifier, schedule = module.functions_for(radius, 25, "common", L=64)
        sch = schedule("CN", .01, "prod")
        sch.update(gtol=jnp.float64(1e-4), tolf=jnp.float64(1e-9))
        name = "POD512_none_CN_25steps_g1e-4"
        lattice, pipeline = module.H, module.P
        rom=module.R
    assert Path(lattice.__file__).resolve() == root / "jcp-hr-baselines" / "hrcore.py"
    warmed = False

    def run(u0, nu, burn_seconds=.1):
        nonlocal warmed
        expected_size = (255 ** 2) if dim == 2 else (63 ** 3)
        host = np.asarray(u0)
        if host.dtype != np.float64 or host.size != expected_size or not np.isfinite(host).all():
            raise ValueError("Expected a finite float64 native interior initial field.")
        initial = jax.block_until_ready(jnp.asarray(host.reshape(-1)))
        visc = jax.block_until_ready(jnp.float64(nu))
        if not warmed:
            jax.block_until_ready(query(initial, visc, data, sch))
            warmed = True
        pipeline.burn(burn_seconds)
        start = time.perf_counter()
        raw = jax.block_until_ready(query(initial, visc, data, sch))
        elapsed = time.perf_counter() - start
        fields = np.asarray(raw["fields"])
        assert fields.dtype == np.float64 and fields.shape == (6, expected_size)
        n = int(sch["steps"])
        gamma=certifier(raw["C"],visc,data,sch)
        witness=rom.certify(raw,gamma,sch,level="prod")
        return {
            "qualification":witness,
            "subject": name,
            "seconds": elapsed,
            "fields": lattice.to_lattice(fields, dim),
            "finite": bool(np.isfinite(fields).all()),
            "reasons": np.asarray(raw["stats"]["reason"])[:n],
            "iterations": np.asarray(raw["stats"]["it"])[:n],
            "saved_coefficients": np.asarray(raw["W"]),
        }

    def initial_from_descriptors(descriptors, index):
        if dim == 2:
            desc = descriptors[index]
            return np.asarray(lattice.E2.initial(256, desc), float)[1:-1, 1:-1].ravel(), float(desc[4])
        return np.asarray(lattice.C3.initial_interior(65, descriptors, index), float), float(descriptors["nu"][index])

    return name, initial_from_descriptors, run
