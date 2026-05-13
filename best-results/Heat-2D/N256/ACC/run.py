#!/usr/bin/env python3
"""
run.py — experiment driver. Calls experiment.train() then experiment.benchmark()
and prints a summary block that the agent greps for metrics.

Do NOT modify unless fixing a demonstrable bug in the driver itself.
"""
from __future__ import annotations

import sys
import time
import traceback

import fixed
import experiment


def main():
    t_start = time.perf_counter()
    try:
        hp = experiment.Hyperparams()
        print(f'hyperparams: {hp}')
        val_trajs = fixed.load_val_trajectories(hp.N, n_test=10)
        print(f'loaded {len(val_trajs)} val trajectories at N={hp.N}')

        t_train = time.perf_counter()
        trained = experiment.train(hp)
        train_seconds = time.perf_counter() - t_train
        print(f'training_seconds: {train_seconds:.1f}')

        t_bench = time.perf_counter()
        metrics = experiment.benchmark(trained, hp, val_trajs)
        bench_seconds = time.perf_counter() - t_bench
        print(f'benchmark_seconds: {bench_seconds:.1f}')

        total_seconds = time.perf_counter() - t_start
        peak_vram_mb = trained.get('peak_vram_mb', 0.0)
        n_params = trained.get('n_params', 0)

        # Summary block in a karpathy-style format -- the agent greps these lines.
        print('---')
        print(f'val_rel_l2:        {metrics["val_rel_l2"]:.6e}')
        print(f'speedup:           {metrics["speedup"]:.4f}')
        print(f'fom_seconds:       {metrics["fom_seconds"]:.4f}')
        print(f'rom_seconds:       {metrics["rom_seconds"]:.4f}')
        print(f'gn_iters_mean:     {metrics["gn_iters_mean"]:.2f}')
        print(f'training_seconds:  {train_seconds:.1f}')
        print(f'benchmark_seconds: {bench_seconds:.1f}')
        print(f'total_seconds:     {total_seconds:.1f}')
        print(f'peak_vram_mb:      {peak_vram_mb:.1f}')
        print(f'n_params:          {n_params}')
        print(f'N:                 {hp.N}')
        print(f'k_dim:             {hp.k_dim}')
        print(f'rank:              {hp.rank}')
        print(f'num_epochs:        {hp.num_epochs}')
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
