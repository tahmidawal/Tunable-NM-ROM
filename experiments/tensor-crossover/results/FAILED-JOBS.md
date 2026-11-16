# Failed jobs (accounted)

| job | job id | elapsed | cause | fix |
|---|---|---|---|---|
| c2d256 | 7250287 | 5 s | vendored checkpoint `*.pkl` was excluded by `.gitignore`, so `git archive` did not stage it | force-added the vendored `.pkl`/`.npz` inputs; resubmitted as c2d256b |
| c3d65 | 7250288 | 5 s | same (`bank.pkl`, `rules3d.npz`) | same; resubmitted as c3d65b |
