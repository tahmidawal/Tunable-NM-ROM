"""Write the exact coordinate-ROM cells for the frozen k/M/m sweep."""
from __future__ import annotations

import json
import os
import sys

KS = (4, 6, 8, 12, 16, 24, 32, 48, 64)
TAUS = (1e-3, 1e-2)


def rows(pde: str) -> list[dict]:
    return [
        dict(pde=pde, method="coord", N=64, k=k, M=4 * k, m=16 * k,
             tau=tau, arm="eq4m")
        for k in KS
        for tau in TAUS
    ]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: make_configs.py <output-directory>")
    out = os.path.abspath(sys.argv[1])
    os.makedirs(out, exist_ok=True)
    for pde in ("poisson2d", "burgers2d"):
        path = os.path.join(out, f"{pde}_configs.json")
        with open(path, "w") as f:
            json.dump(rows(pde), f, indent=1)
            f.write("\n")
        print(path, flush=True)


if __name__ == "__main__":
    main()
