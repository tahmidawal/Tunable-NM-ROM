#!/usr/bin/env python3
"""
run.py — Heat-3D autoresearch driver.
Do NOT modify unless fixing a demonstrable bug.
"""
from __future__ import annotations

import sys
import os
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment


def main():
    t_start = time.perf_counter()
    try:
        metrics = experiment.run_and_benchmark()
        total_seconds = time.perf_counter() - t_start

        print('---')
        print(f'val_rel_l2:        {metrics["val_rel_l2"]:.6e}')
        print(f'speedup:           {metrics["speedup"]:.4f}')
        print(f'fom_seconds:       {metrics["fom_seconds"]:.4f}')
        print(f'rom_seconds:       {metrics["rom_seconds"]:.4f}')
        print(f'gn_iters_mean:     {metrics["gn_iters_mean"]:.2f}')
        print(f'training_seconds:  {metrics.get("training_seconds", 0.0):.1f}')
        print(f'total_seconds:     {total_seconds:.1f}')
        print(f'peak_vram_mb:      {metrics.get("peak_vram_mb", 0.0):.1f}')
        print(f'n_params:          {metrics.get("n_params", 0)}')
        print(f'N:                 {metrics.get("N", 0)}')
        print(f'k_dim:             {metrics.get("k_dim", 0)}')
        print(f'rank:              {metrics.get("rank", 0)}')
        print(f'n_eq:              {metrics.get("n_eq", 0)}')
        print(f'status:            ok')
    except Exception:
        traceback.print_exc()
        total_seconds = time.perf_counter() - t_start
        print('---')
        print(f'val_rel_l2:        nan')
        print(f'speedup:           0.0')
        print(f'total_seconds:     {total_seconds:.1f}')
        print(f'peak_vram_mb:      0.0')
        print(f'status:            crash')
        sys.exit(1)


if __name__ == '__main__':
    main()
