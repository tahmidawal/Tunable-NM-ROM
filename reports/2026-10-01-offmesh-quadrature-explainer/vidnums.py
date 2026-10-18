"""Every number shown in the video, with the report snippet it comes from.

`check()` asserts each snippet occurs verbatim in its source report and that every
value string occurs inside its snippet, so the video cannot drift from the reports.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]  # reports/
R3 = REPO / "2026-10-01-burgers3d-offmesh-quadrature.md"
R2 = REPO / "2026-10-01-burgers2d-offmesh-quadrature.md"

# key: (values shown, source file, verbatim snippet containing them)
SRC = {
    # 3D, R'=512, worst continuum rho at 64^3 (section 3 table)
    "rho3d_lat": (["8.3e-02", "1.0e-02", "1.1e-03", "8.6e-05"], R3,
                  None),
    "rho3d_gl": (["6.2e-01", "1.0e-02", "4.7e-04"], R3, None),
    "rho3d_sob": (["1.5e-01"], R3, "| `sob16384` | 16384 | 1.5e-01 (1.1e-01)"),
    "rho3d_smol": (["4.3e+01"], R3, "| `smol8` | 2559 | 4.3e+01 (1.3e+01)"),
    "bar": (["0.116"], R3, "crosses the 0.116 bar at 256³"),
    # mesh gap: dense upwind vs continuum, R'=512
    "gap": (["1.9e-01", "9.8e-02", "5.2e-02"], R3,
            "| dense sign-upwind | mesh | 1.9e-01 (1.3e-01) | 9.8e-02 (6.6e-02) | 5.2e-02 (3.3e-02)"),
    # held-out worst refined error, R'=512
    "err_tensor": (["10.46", "6.09", "3.87"], R3,
                   "the tensor at $R' = 512$ is at 10.46 % (64³), 6.09 % (128³), 3.87 % (256³)"),
    "err_fom": (["9.85", "5.22", "2.27"], R3, "whose best setting is at 9.85 %, 5.22 %, 2.27 %"),
    "err_off": (["2.83"], R3, "The selected off-mesh rule is at 2.83 %, 2.83 %, 2.83 %"),
    # cost, R'=512
    "ms": (["47.9", "34.1", "52.0", "40.6", "90.1", "76.7"], R3,
           "64³/512: 34.1 vs 47.9 ms; 64³/256: 11.1 vs 12.2 ms; 128³/512: 40.6 vs 52.0 ms; 128³/256: 12.1 vs 14.1 ms; 256³/512: 76.7 vs 90.1 ms"),
    "mem": (["4.30 GB", "0.34 GB"], R3, None),
    "speedup": (["2.21×", "2.59×", "6.45×"], R3,
                "the 256³ speedups move from 2.21× (tensor) to 2.59× (selected), $R' = 512$; 5.66× (tensor) to 6.01× (selected), $R' = 256$; under the refined rule they move from 2.21× to 6.45×"),
    "bank129": (["129"], R3, "The bank was trained on fields up to 129 nodes per axis"),
    # 2D test cohort
    "err2d": (["2.64", "5.58"], R2,
              "acc: continuum rollout ST 2.64/2.65/1.57 %, S 1.34/1.36/0.10 % at $256^2/1024^2/4096^2$; dense ST 5.58/3.27/1.62 %"),
    "spread2d": (["0.011 pp"], R2, "0.011 pp for the continuum rollout"),
    "t2d_acc": (["55.9", "55.5", "55.4"], R2, "| acc | Gauss $96^2$ | 55.9 | 55.5 | 55.4 | 0.991 | yes |"),
    "t2d_fast": (["13.3", "13.0", "13.4"], R2, "| fast | Fibonacci 1597 | 13.3 | 13.0 | 13.4 | 1.008 | yes |"),
    "t2d_lat": (["47.7", "46.9", "46.6"], R2, "| acc | mesh lattice $63^2$ | 47.7 | 46.9 | 46.6 | 0.976 | yes |"),
    "b4": (["0.972", "1.040"], R2, "between 0.972 and 1.040 for all 40 measured arms"),
    "head2d": (["0.585", "1.105", "1.258"], REPO.parent / "LAB-LOG.md",
               "B3 0.585/1.105/1.258 % vs 0.5 %"),
}

ROW_3D = {  # section-3 R'=512 rows (first row match), for the rho tables
    "rho3d_lat": ["| `lat4096` | 4096 | 8.3e-02", "| `lat8192` | 8192 | 1.0e-02",
                  "| `lat16384` | 16384 | 1.1e-03", "| `lat32768` | 32768 | 8.6e-05"],
    "rho3d_gl": ["| `gl16` | 4096 | 6.2e-01", "| `gl24` | 13824 | 1.0e-02", "| `gl32` | 32768 | 4.7e-04"],
    "mem": ["| 64³ | 512 | `tensor` | 4.30 GB", "| 64³ | 512 | `gl24` (selected) | 0.34 GB"],
}


def v(key, i=0):
    return SRC[key][0][i]


def check():
    for key, (vals, path, snip) in SRC.items():
        text = path.read_text()
        snips = [snip] if snip else ROW_3D[key]
        for s in snips:
            assert s in text, f"{key}: snippet not in {path.name}: {s!r}"
        joined = " ".join(snips)
        for val in vals:
            assert val in joined, f"{key}: value {val} not in its snippet"
    print(f"numbers.check: {len(SRC)} keys verified against the reports")


if __name__ == "__main__":
    check()
