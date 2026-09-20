"""Bounded reused-fixture panel for efficient CG and its traced control."""
import json
from pathlib import Path
import run
import audit

source=Path(__file__).parent/'runs/cg-history-smoke'
out=Path(__file__).parent/'runs/cg-plain-smoke';assert not out.exists()
cfg=json.loads((source/'result.json').read_text())['config']
cfg.update(reuse_checkpoint_directory=str(source),plain_cg_control=True)
run.run(cfg,out,smoke=True)
audit.audit(out,out/'audit.json')
