"""One-off extraction of J2's log-only timers into runs/j2/pulled/output/probe_numbers.json (input of analysis.py)."""
import json
from pathlib import Path

import analysis as AN

HERE = Path(__file__).resolve().parent
out = HERE / 'runs' / 'j2' / 'pulled' / 'output' / 'probe_numbers.json'
d = AN.probe_from_logs(HERE / 'runs')
d['extracted_from'] = ['runs/j2/pulled/output/r3a.log', 'runs/j2/pulled/logs/*.out']
out.write_text(json.dumps(d, indent=1))
print(out)
