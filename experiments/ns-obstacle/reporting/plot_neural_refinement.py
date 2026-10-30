"""Recorded accepted-state warm refinement; training metrics are not flow errors."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser(description=__doc__)
for name in ('metadata', 'review', 'output'):
    p.add_argument('--'+name, type=Path, required=True)
a = p.parse_args()
review = json.loads(a.review.read_text())
initial = json.loads((a.metadata/'state-00000-initial.json').read_text())
accepted = [json.loads(line) for line in (a.metadata/'ACCEPTED.jsonl').read_text().splitlines()]
assert review['passed'] and len(accepted) == review['new_accepted_lbfgs_steps']
assert accepted[-1]['full_post_update_objective'] == review['selected']['objective']
steps = np.array([0]+[r['iteration'] for r in accepted])
loss = np.array([initial['objective']]+[r['full_post_update_objective'] for r in accepted])
metrics = [initial['metrics']]+[r['metrics'] for r in accepted]
v = 100*np.sqrt([r['value_relative_squared_by_mode'] for r in metrics])
g = 100*np.sqrt([r['gradient_relative_squared_by_mode'] for r in metrics])
assert np.isfinite(loss).all() and np.isfinite(v).all() and np.isfinite(g).all()
assert np.array_equal(steps, np.arange(len(steps)))
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout='constrained')
axes[0].semilogy(steps, loss, color='#244870')
axes[0].scatter([steps[0], steps[-1]], [loss[0], loss[-1]], color='#d46332', zorder=5)
axes[0].set(title='Unchanged training objective', ylabel='Weighted relative squared errors')
for i in range(v.shape[1]):
    axes[1].semilogy(steps, v[:, i], label=f'Mode {i+1}')
    axes[2].semilogy(steps, g[:, i], label=f'Mode {i+1}')
axes[1].set(title='Velocity-mode fitting error', ylabel='Relative mode error [%]')
axes[2].set(title='Gradient-mode fitting error', ylabel='Relative mode error [%]')
axes[2].legend(ncol=2)
for ax in axes:
    ax.set_xlabel('Additional accepted L-BFGS steps')
    ax.grid(alpha=.2)
fig.suptitle('Longer neural-bank refinement: recorded training diagnostics', fontsize=15)
fig.supxlabel(f"{len(accepted):,} additional accepted steps; no new Adam updates. Warm restart with reset curvature.\n"
              f"Termination: {review['optimizer_status']}, not convergence. Training-mode errors are not actual ROM flow errors.", fontsize=10)
a.output.mkdir(parents=True, exist_ok=True)
for ext in ('png', 'pdf'):
    fig.savefig(a.output/('training-progress.'+ext), dpi=150)
plt.close(fig)
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
files = ['state-00000-initial.json', 'ACCEPTED.jsonl', 'CONFIG.json', 'FINAL.json', 'MODEL-RECEIPT.json']
(a.output/'PLOT-MANIFEST.json').write_text(json.dumps({
    'generator_sha256': sha(Path(__file__)), 'review_sha256': sha(a.review),
    'inputs': {name: sha(a.metadata/name) for name in files},
    'model_sha256': review['model_sha256'], 'selected': review['selected'],
    'scope': 'Accepted-state training history only. No timing comparison or inferred flow accuracy.'
}, indent=2)+'\n')
