#!/usr/bin/env python
"""Estimate parameter counts for the mirrored-ViT sweep configs.

Pure-Python (no jax/flax), so it can run on the login node. Useful to
sanity-check that no config will OOM the A100 before we submit it.
"""
from __future__ import annotations

import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def vit_block_params(d_emb: int, mlp_ratio: float = 4.0) -> int:
    # 2 LayerNorm (2*d), self-attn qkv+out (4 * d*d), MLP (2 * d * mlp*d).
    h = int(d_emb * mlp_ratio)
    return 2 * (2 * d_emb) + 4 * d_emb * d_emb + (d_emb * h + h * d_emb)


def encoder_params(N: int, d: int, P: int, d_emb: int, L: int, k: int) -> int:
    n_per_side = N // P
    num_patches = n_per_side ** d
    patch_feat = P ** d
    p = 0
    p += patch_feat * d_emb + d_emb               # patch_embed Dense
    p += num_patches * d_emb                       # pos embedding
    p += L * vit_block_params(d_emb)
    p += 2 * d_emb                                 # final LayerNorm
    p += d_emb * k + k                             # head Dense -> latent
    return p


def vit_decoder_params(N: int, d: int, P: int, d_emb: int, L: int, k: int,
                        include_linear_skip: bool = False) -> int:
    n_per_side = N // P
    num_patches = n_per_side ** d
    patch_feat = P ** d
    num_nodes = N ** d
    p = 0
    p += k * (num_patches * d_emb) + num_patches * d_emb   # latent -> tokens Dense
    p += num_patches * d_emb                                # pos embedding
    p += L * vit_block_params(d_emb)
    p += 2 * d_emb                                          # final LayerNorm
    p += d_emb * patch_feat + patch_feat                    # patch head Dense
    p += 1                                                  # bias scalar
    if include_linear_skip:
        p += k * num_nodes                                  # W_direct
    return p


def total_for_config(cfg: dict, has_skip: bool) -> dict:
    N = cfg["N"]; d = cfg["spatial_dim"]
    P = cfg["patch_size"]
    d_emb_enc = cfg["embed_dim"]; L_enc = cfg["num_enc_layers"]; k = cfg["latent_dim"]
    dec_P = cfg.get("dec_patch_size") or P
    dec_emb = cfg.get("dec_embed_dim") or d_emb_enc
    dec_L = cfg.get("dec_num_layers") or L_enc
    enc = encoder_params(N, d, P, d_emb_enc, L_enc, k)
    dec = vit_decoder_params(N, d, dec_P, dec_emb, dec_L, k, include_linear_skip=has_skip)
    return {"enc": enc, "dec": dec, "total": enc + dec,
            "skip_params": k * (N ** d) if has_skip else 0}


def main():
    for pkg, has_skip in [("poisson", True), ("heat", False)]:
        print(f"\n=== {pkg.upper()} (linear_skip={has_skip}) ===")
        cfg_dir = ROOT / pkg / "configs"
        for cfg_path in sorted(cfg_dir.glob("*.yaml")):
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            cnts = total_for_config(cfg, has_skip)
            print(f"  {cfg_path.name:<40} enc={cnts['enc']/1e6:>6.2f}M  "
                  f"dec={cnts['dec']/1e6:>6.2f}M  total={cnts['total']/1e6:>6.2f}M"
                  + (f"  skip={cnts['skip_params']/1e6:.2f}M" if has_skip else ""))


if __name__ == "__main__":
    main()
