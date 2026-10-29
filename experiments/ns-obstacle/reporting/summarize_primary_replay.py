"""Extract measured native replay evidence without extrapolating to a finer mesh."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
d=json.loads(a.input.read_text());assert d['passed'] and not d['physical_admission'] and not d['training_released']
keys=['passed','physical_admission','training_released','scope','counts','original_volume_seconds','primary_capture_seconds','restoration','replay_accounting','all_array_comparisons_exact','primary_directory_bytes','original_directory_bytes','elapsed_seconds','peak_rss_bytes','original_manifest_sha256','primary_manifest_sha256']
r={k:d[k] for k in keys};r.update(source_result_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),fine_P4_extrapolation_authorized=False)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n')
