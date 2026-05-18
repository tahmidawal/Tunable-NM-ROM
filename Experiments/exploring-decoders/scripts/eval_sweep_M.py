#!/usr/bin/env python
"""Sweep M (query points per snapshot) and compute rel-L2 + per-query FLOPs.

For each trained decoder and each M in M_LIST, evaluate the AE on the
validation set with that M, then report (mean rel-L2, total FLOPs per
query) so the result can be plotted.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tunable_rom_decoders.autoencoder import INRAutoencoder
from tunable_rom_decoders.training import load_checkpoint, make_val_set
from tunable_rom_decoders.fom.poisson_analytical import PoissonAnalytical
from tunable_rom_decoders.flops import siren_flops, xattn_flops
from tunable_rom_decoders.utils.config import ExperimentConfig


M_LIST_DEFAULT = (16, 64, 256, 1024, 4096, 16384)


def _build_model(cfg_dict: dict) -> INRAutoencoder:
    cfg = ExperimentConfig(**cfg_dict)
    return INRAutoencoder(
        decoder_kind=cfg.decoder_kind,
        N=cfg.N,
        spatial_dim=cfg.spatial_dim,
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        num_enc_layers=cfg.num_enc_layers,
        latent_dim=cfg.latent_dim,
        coord_dim=cfg.spatial_dim,
        hidden_dim=cfg.hidden_dim,
        siren_num_layers=cfg.siren_num_layers,
        omega_0=cfg.omega_0,
        omega=cfg.omega,
        modulator_hidden=cfg.modulator_hidden,
        d_attn=cfg.d_attn,
        num_fourier=cfg.num_fourier,
        xattn_num_layers=cfg.xattn_num_layers,
        fourier_scale=cfg.fourier_scale,
    ), cfg


def _flops_for(cfg: ExperimentConfig, num_tokens: int):
    if cfg.decoder_kind == "siren":
        return siren_flops(
            coord_dim=cfg.spatial_dim,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.siren_hidden_dim,
            num_layers=cfg.siren_num_layers,
            modulator_hidden=cfg.modulator_hidden,
        )
    return xattn_flops(
        coord_dim=cfg.spatial_dim,
        latent_dim=cfg.latent_dim,
        embed_dim=cfg.embed_dim,
        d_attn=cfg.d_attn,
        num_fourier=cfg.num_fourier,
        hidden_dim=cfg.xattn_hidden_dim,
        num_layers=cfg.xattn_num_layers,
        num_tokens=num_tokens,
    )


def evaluate(ckpt_path: Path, data_path: Path, M_list, seed_offset: int = 0):
    obj = load_checkpoint(ckpt_path)
    params = obj["params"]
    cfg_dict = obj["config"]
    model, cfg = _build_model(cfg_dict)

    data = np.load(data_path)
    U_val = data["U_val"]
    freqs_val = data["freqs_val"]

    fom = PoissonAnalytical(N=cfg.N, spatial_dim=cfg.spatial_dim)
    num_tokens = (cfg.N // cfg.patch_size) ** cfg.spatial_dim
    fb = _flops_for(cfg, num_tokens)

    results = []
    for M in M_list:
        x_val, u_val = make_val_set(
            freqs_val, M, cfg.N, cfg.spatial_dim, fom, seed=1234 + seed_offset
        )
        U_val_j = jnp.asarray(U_val)
        x_j = jnp.asarray(x_val)
        u_j = jnp.asarray(u_val)

        def per_sample(u_in, x_q, u_t):
            u_pred = model.apply({"params": params}, u_in, x_q)
            return jnp.linalg.norm(u_pred - u_t) / (jnp.linalg.norm(u_t) + 1e-12)

        rel_l2 = float(jnp.mean(jax.vmap(per_sample)(U_val_j, x_j, u_j)))
        flops = fb.total_per_query(M)
        results.append(
            {"M": int(M), "rel_l2": rel_l2, "flops_per_query": float(flops),
             "amortized": int(fb.amortized), "per_query": int(fb.per_query)}
        )
        print(f"[eval] {cfg.decoder_kind:>5s}  M={M:>6d}  rel_L2={rel_l2:.4e}  "
              f"flops/q={flops:.2e}")
    return {
        "decoder_kind": cfg.decoder_kind,
        "config": cfg_dict,
        "results": results,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", action="append", required=True,
                   help="repeatable: path to a trained checkpoint")
    p.add_argument("--data", required=True, type=Path,
                   help="shared validation data .npz")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--M", type=int, nargs="*", default=list(M_LIST_DEFAULT))
    args = p.parse_args()

    out = {"M_list": args.M, "decoders": []}
    for ckpt_path in args.ckpt:
        print(f"\n=== Evaluating checkpoint {ckpt_path} ===")
        res = evaluate(Path(ckpt_path), args.data, args.M)
        out["decoders"].append({"ckpt": str(ckpt_path), **res})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved sweep results to {args.out}")


if __name__ == "__main__":
    main()
