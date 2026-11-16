"""results/train/crom-<cfg>.json from the CROM training job log lines (exit/seconds, GPU name, last progress bar).
  crom_train_json.py <cfg> <log> <out>"""
import json, re, sys
cfg, log, out = sys.argv[1:4]
t = open(log).read()
gpu = re.search(r'^(NVIDIA [^,]+),', t, re.M).group(1)
sec = int(re.search(r'seconds=(\d+)', t).group(1))
job = re.search(r'job=(\d+)', t).group(1)
loss = float(re.search(r'loss=([0-9.]+)', t).group(1))
steps = int(re.search(r'(\d+)/\1 \[', t).group(1)) * 31
json.dump(dict(name=f'CROM {cfg}', steps=steps, gpu=gpu, train_seconds=sec, job_id=job, loss_last_10pct=loss,
               notes='31 epochs, strict float32 (A1); loss = Lightning progress-bar running loss at the last step (x1000 MSE)'), open(out, 'w'), indent=1)
print(open(out).read())
