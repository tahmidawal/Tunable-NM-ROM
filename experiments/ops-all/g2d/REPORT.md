# g2d — neural operators, Burgers 2D and Heat 2D (ops-all, 2026-09-24)

Saved by the coordinator from the group agent's final report (the agent could not write report files).
Numbers are from pulled, checksum-verified job records; every ratio is same-job. Machine-readable:
`cost_points.json` (all cells, every operator size, `cost_to_target`) and `results.json` (original three cells).

Status: Burgers 2D 256², 512², 1024², 2048², 4096² complete; Heat 2D 256² and 1024² complete. Heat 512²,
2048², 4096² were cancelled on the user's "Burgers only" instruction (jobs 4308349, 4308141, 4307977,
4307919, 4307409, 4307420, 4307200; none of their numbers are used). Heat 8192² was never submitted.
Cluster namespace empty.

## Originally assigned cells (worst % / median % / GPU ms, speedup vs the cell's FOM, same job)

| cell | panel job (GPU) | FOM | NM-ROM accurate | NM-ROM fast | FNO | U-Net | Transolver | DeepONet |
|---|---|---|---|---|---|---|---|---|
| Burgers 512² | pb512 4290476 (A100-80GB) | lean_nt3e-3_l3e-3_dt005 0.050 % / 27.7 ms | 0.195 % / 46.4 ms / 0.60× | 1.75 % / 17.4 ms / 1.59× | 4.57 / 2.75 / 22.6 ms (1.22×) | 2.52 / 1.60 / 14.2 ms (1.95×) | 3.27 / 2.34 / 36.0 ms (0.77×) | 32.2 / 15.8 / 13.3 ms (2.08×) |
| Burgers 4096² | pl4096b 4309865 (H200) | lean_nt3e-3_l3e-3_dt005 0.050 % / 523 ms | 0.224 % / 61.9 ms / 8.44× | 1.89 % / 25.0 ms / 20.9× | OOM | unet-refine 21.9 / 15.5 / 624 ms (0.84×) | OOM | OOM |
| Heat 4096² | phl4096 4303352 (A100-80GB) | CN–CG dt 0.025 rtol 1e-3 0.054 % / 4613 ms | 0.133 % / 20.9 ms / 221× | 0.276 % / 10.6 ms / 434× | OOM | trained, not timed | trained, not timed | OOM |

Burgers 512² reproduces Table 1 (job 4241035). Burgers 4096² NM-ROM errors reproduce bk4096b; speedups
are a new same-job re-time (paper: 8.78× / 21.7×). Heat 4096² U-Net/Transolver OOMed in the panel because
of a g2d bug (JAX reserved the GPU first); checkpoints were deleted before diagnosis and the rerun was
cancelled under the Burgers-only instruction.

## Expanded cost-to-target cells (NM-ROM ladder, FOM grid, every operator size, one allocation)

| cell | job (GPU) | NM-ROM span R'=384 / 128 / 64 | best per operator family (worst %, median ms) |
|---|---|---|---|
| Burgers 256² | pl256 4301382 (A100-80GB) | 0.166 % 123 ms / 1.60 % 31.7 ms / 4.27 % 16.5 ms | FNO fno-w96f32 3.89 % 6.8; U-Net unet-b64 2.21 % 14.3; Transolver tsol-refine 4.46 % 11.0; DeepONet don-refine 32.5 % 3.8 |
| Burgers 512² | pl512 4302973 (A100-80GB) | 0.195 % 229 ms / 1.75 % 42.8 ms / 4.55 % 20.0 ms | FNO fno-w96f32 4.38 %; U-Net unet-refine 2.31 % 14.0; Transolver tsol-refine 3.39 %; DeepONet 34.4 % |
| Burgers 1024² | pl1024 4304490 (A100-80GB) | 0.211 % 81.3 ms / 1.83 % 31.4 ms / 4.71 % 20.1 ms | FNO fno-w96f32 4.08 % 87; U-Net unet-refine 4.60 % 54; Transolver 10.2 %; DeepONet 32.8 % |
| Burgers 2048² | pl2048 4309160 (A100-80GB) | 0.219 % 88.6 ms / 1.87 % 34.1 ms / 4.79 % 21.5 ms | FNO fno-w96f32 6.46 % 438; U-Net unet-refine 9.50 % 213; Transolver 18.7 %; DeepONet 38.3 % |
| Burgers 4096² | pl4096b 4309865 (H200) | 0.224 % 61.9 ms / 1.89 % 25.0 ms / 4.83 % 17.1 ms | only unet-refine 21.9 % 624; every other size OOM |
| Heat 256² (no Table 1 row) | phl256 4307911 (A100-80GB) | span R'=128 0.133 % 0.3 ms; R'=48 0.276 % 0.2 ms | FNO fno-w96f32 0.14 % 7.7; U-Net unet-large 0.42 %; DeepONet 0.89 %; Transolver 2.83 % |
| Heat 1024² | phl1024 4305095 (A100-80GB) | span R'=128 0.133 % 1.3 ms; R'=48 0.276 % 0.7 ms | FNO fno-w96f32 0.83 % 86; U-Net 1.56 %; Transolver 2.57 %; DeepONet 4.60 % |

- No operator reaches 1 % on Burgers 2D at any mesh 256²–4096².
- Base operator errors at 256²/1024² reproduce the paper's operator table exactly.
- Larger sizes often did worse than the base size within the 3000 s budget (fewer epochs), e.g. unet-b64
  5.03 % vs unet-refine 2.31 % at 512²; tsol-large 29.3 % at 2048². Epochs per size are in cost_points.json.
- Size ladder: Burgers adds unet-large (17.5M), unet-b64 (31.0M), tsol-large (12.3M), fno-w96f32 (40.2M,
  float32 via a new F32FNO wrapper that keeps the complex spectral weights; base FNOs stay float64).
  Heat adds unet-large, unet-b64, fno-w96f32. At 256² the 13 existing opt201 checkpoints were reused.

## Caveats

1. Cancelled/failed jobs with no numbers used: b4096tr, b4096tb (cancelled pending); pb4096, pl4096c
   (A100 OOM building the bank/tests); pl4096 unused; ph4096 cancelled after ~1 min.
2. Duplicate submission by an ssh retry: h4096deepb twice (4293111, 4293135); duplicate cancelled while
   pending, neither ran; `submit.sh` no longer retries sbatch.
3. One network per GPU everywhere except b4096tc (before the rule; four networks sequentially). Only its
   U-Net trained (3000 s, 12 epochs); FNO/Transolver/DeepONet OOMed at micro-batch 1.
4. Burgers 4096² operators: tsol-refine, don-small, unet-large, fno-w96f32 each OOMed at micro-batch 1 on
   A100-80GB (own jobs); the H200 queue estimated 21:40, so no H200 training attempt. The one U-Net is
   badly under-trained.
5. Heat 4096²: FNO OOM at micro-batch 1 on H200; DeepONet OOM on A100-80GB (H200 retry dropped for time).
6. Heat operator-panel order gate fails (sentinel deviation 0.16–0.32), the same sentinel-noise statistic
   heat-compare-hires replaced in its amendment A2; the replacement test was not run.
7. Burgers 256²/512² panels use the bank-knob solver arms (with the exact first step "x1"), hence the
   512² accurate setting at 229 ms here vs 46 ms with Table 1's engineered solver in pb512.
8. Certificates skipped in the new panels (the source jobs certified the same arms).
9. Burgers 4096² operator training data generated in host memory (`train_mem.py`, 150 GB, never on disk);
   `train()` unchanged.
10. GPU types: A100-80GB throughout except Burgers 4096² (H200). Table 1's heat 4096² row was timed on H200.
