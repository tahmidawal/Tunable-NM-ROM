"""jcp-robust E2 3D driver (DESIGN 3.3; A1.7-A1.9, A2.6, A3.5, A3.6): jcp-dims's d3run (copied unchanged) on a
retrained W1024-recipe bank. The config's "bank_registry" {name: {"path", "sha256"}} adds bank entries to c3common.BANKS
(sha256 verified by c3common.load_bank as for the incumbent); settings name these banks. Blocks are d3run's (ladder for
the selection ladders, panel for the front timing). Nothing else differs from d3run.

    python e2run3d.py --config configs/<attempt>.json --out <dir>
"""
import json
import sys
from pathlib import Path

import c3common as C3

HERE = Path(__file__).resolve().parent


def install_banks(cfg):
    for name, b in (cfg.get('bank_registry') or {}).items():
        assert name not in C3.BANKS, f'bank name {name} would shadow an existing bank'
        C3.BANKS[name] = (HERE / b['path'], b['sha256'])


if __name__ == '__main__':
    args = sys.argv[1:]
    cfg = json.loads(Path(args[args.index('--config') + 1]).read_text())
    install_banks(cfg)
    import d3run
    d3run.main()
