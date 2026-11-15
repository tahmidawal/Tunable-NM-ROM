# GPU ledger (linear-query-speed)

| job | GPU | purpose | wall | GPU-h | state |
|---|---|---|---|---|---|
| 7220222 | H200 (requested) | selection only (superseded) | 0 | 0 | cancelled while pending, never started |
| 7220223 | H200 (requested) | paper driver only (superseded) | 0 | 0 | cancelled while pending, never started |
| 7220234 | H200 (pax009) | dev selection 256^2, 4096^2 -> freeze -> test 256^2..4096^2; then the unmodified paper driver at 256^2 and 1024^2 (its own timing-stability 'neighbour' gate failed at both, ratios 1.185 and 1.105 against a limit of 1.1; errors match stored) | 01:36:44 | 1.61 | completed |
| 7220235 | A100 80GB (pax050) | dev selection 256^2, 2048^2 -> freeze -> test 256^2..2048^2; 4096^2 test failed (compression out of device memory) | 01:06:10 | 1.10 | completed |
| 7221131 | A100 80GB (pax106) | test 4096^2 only, frozen pair of 7220235, `--host-compress` (amendment A2) | 00:24:15 | 0.40 | completed |
| 7221749 | H200 (requested) | heat 2D with the wrong fast arm (see A3a) | 0 | 0 | cancelled while pending, never started |
| 7221755 | H200 (requested) | heat 2D: selection at 4096^2 on the development cohort, freeze, test 4096^2/2048^2/1024^2 (A3, A3a) | 0 so far | 0 | still pending on priority at 20:30 EST |
| 7228875 | A100 80GB (requested) | heat 2D fallback: selection at 2048^2 on the development cohort, freeze, test 2048^2/1024^2 (A3b) | 00:08:57 | 0.15 | completed (pax106) |
| 7228913 | A100 80GB (requested) | heat 2D 4096^2 with the frozen file of 7228875 (A3c) | 00:05:22 | 0.09 | completed (pax106) |

Total GPU-hours used: 3.35 (1.61 + 1.10 + 0.40 + 0.15 + 0.09), one GPU per job.
