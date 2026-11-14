# GPU ledger (linear-query-speed)

| job | GPU | purpose | wall | GPU-h | state |
|---|---|---|---|---|---|
| 7220222 | H200 (requested) | selection only (superseded) | 0 | 0 | cancelled while pending, never started |
| 7220223 | H200 (requested) | paper driver only (superseded) | 0 | 0 | cancelled while pending, never started |
| 7220235 | A100 80GB (pax050) | dev selection 256^2, 2048^2 -> freeze -> test 256^2..2048^2; 4096^2 test failed (compression out of device memory) | 01:06:10 | 1.10 | completed |
| 7221131 | A100 80GB (pax106) | test 4096^2 only, frozen pair of 7220235, `--host-compress` (amendment A2) | 00:24:15 | 0.40 | completed |
| 7221749 | H200 (requested) | heat 2D with the wrong fast arm (see A3a) | 0 | 0 | cancelled while pending, never started |
| 7221755 | H200 (requested) | heat 2D: selection at 4096^2 on the development cohort, freeze, test 4096^2/2048^2/1024^2 (A3, A3a) | pending | | |
