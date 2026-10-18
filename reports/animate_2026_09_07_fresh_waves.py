"""Animate audited actual reference/ROM observations, without interpolating time."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from PIL import Image
from plot_2026_09_07_fresh_wave_fields import checked_inputs, state_metrics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def palette():
    colors = (plt.get_cmap('RdBu_r')(np.linspace(0, 1, 240))[:, :3]*255).astype(np.uint8).tolist()
    colors += [[i, i, i] for i in np.linspace(0, 255, 16).astype(int)]
    image = Image.new('P', (1, 1))
    image.putpalette(np.asarray(colors, dtype=np.uint8).ravel().tolist())
    return image


def render(out, boundary, view, times, reference, decoder, n, rank, seed):
    reflective = boundary=='reflective'
    def complete(values):
        return np.pad(values, ((0, 0), (1, 1), (1, 1))) if reflective else values
    arrays = [complete(reference), complete(decoder)]
    extent = float(max(np.max(abs(x)) for x in arrays))
    norm = Normalize(-extent, extent)
    surface = view=='surface'
    figure = plt.figure(figsize=(10.4, 5.0), dpi=90)
    axes = [figure.add_subplot(1, 2, i+1, projection='3d' if surface else None) for i in range(2)]
    figure.subplots_adjust(left=.035, right=.91, top=.84, bottom=.1, wspace=.11)
    title = 'Reflective waves' if reflective else 'Absorbing waves'
    figure.suptitle(title+' — wave displacement evolving over time', fontsize=15, y=.97)
    stamp = figure.text(.5, .88, '', ha='center', fontsize=13)
    figure.text(.5, .025, 'Fixed amplitude and color scale throughout • reference left, decoder right', ha='center', fontsize=10)
    cmap = plt.get_cmap('RdBu_r')
    axis = np.linspace(0, 1, n+1)
    x, y = np.meshgrid(axis[::4], axis[::4], indexing='ij')
    for i, ax in enumerate(axes):
        ax.set_title('Reference solution' if i==0 else f'MLP decoder ({rank} coordinates)', fontsize=12, pad=6)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='x', ylabel='y', xticks=[0, .5, 1], yticks=[0, .5, 1])
        ax.tick_params(labelsize=8)
        if surface:
            ax.set(zlim=(-extent, extent), zlabel='u', zticks=[-extent, 0, extent])
            ax.view_init(elev=27, azim=-57)
            ax.set_box_aspect((1, 1, .55))
        else:
            ax.set_aspect('equal')
    colorbar = figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=figure.add_axes([.94, .2, .015, .5]))
    colorbar.set_label('Displacement u')
    frames, payloads, artists = [], [], [None, None]
    pal = palette()
    for index, time in enumerate(times):
        stamp.set_text(f't = {time:.2f}')
        for i, ax in enumerate(axes):
            value = arrays[i][index]
            if surface:
                if artists[i] is not None:
                    artists[i].remove()
                artists[i] = ax.plot_surface(x, y, value[::4, ::4], cmap=cmap, norm=norm, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
            elif artists[i] is None:
                artists[i] = ax.imshow(value.T, origin='lower', extent=(0, 1, 0, 1), cmap=cmap, norm=norm, interpolation='nearest')
            else:
                artists[i].set_data(value.T)
        figure.canvas.draw()
        rgb = Image.fromarray(np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy())
        frame = rgb.quantize(palette=pal, dither=Image.Dither.NONE)
        frames.append(frame)
        buffer = io.BytesIO()
        frame.save(buffer, format='PNG')
        payloads.append('data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode())
        if index in (0, len(times)//2, len(times)-1):
            rgb.save(out/f'{boundary}-{view}-frame-{index:03}.png')
    plt.close(figure)
    filename = f'2026-09-07-{boundary}-wave-{view}.gif'
    durations = [100]*len(frames)
    durations[0], durations[-1] = 400, 900
    frames[0].save(out/filename, save_all=True, append_images=frames[1:], duration=durations, loop=0, disposal=2, optimize=False)
    with Image.open(out/filename) as saved:
        assert saved.n_frames==len(times)
        decoded_last = None
        for index in range(saved.n_frames):
            saved.seek(index)
            assert saved.size==frames[0].size
            decoded_last = saved.convert('RGB')
        assert decoded_last is not None and decoded_last.tobytes()==frames[-1].convert('RGB').tobytes()
    record = dict(file=filename, frames=len(times), frame_duration_ms=durations, fixed_color_limits=[-extent, extent], fixed_height_limits=[-extent, extent] if surface else None, render_spatial_stride=4 if surface else 1, source_optimizer_seed=seed, sha256=sha(out/filename))
    print(boundary, view, 'rendered', len(times), 'frames', flush=True)
    return record, payloads


HTML = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Wave evolution</title><style>body{margin:24px auto;padding:0 18px;max-width:1120px;font:16px system-ui;color:#182233;background:#f7f9fc}h1{margin-bottom:8px}p{color:#455368}.controls{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin:18px 0}button,select{font:inherit;padding:8px 12px;border:1px solid #bac5d4;border-radius:6px;background:white}button{cursor:pointer}img{width:100%;height:auto;background:white;border:1px solid #dae1eb;border-radius:9px}input{flex:1;min-width:180px}output{min-width:90px}small{display:block;margin-top:12px;color:#526175}</style></head><body><h1>Wave evolution</h1><p>Watch the reference solution and the decoder evolve from the same initial pulse.</p><div class="controls"><label>Boundary <select id="boundary"><option value="reflective">Reflective</option><option value="absorbing">Absorbing</option></select></label><label>View <select id="view"><option value="surface">Surface</option><option value="top">Top view</option></select></label><button id="play">Pause</button><button id="restart">Restart</button><label>Speed <select id="speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label></div><img id="movie" alt="Reference wave on the left and decoder prediction on the right"><div class="controls"><input id="time" type="range" min="0" max="48" value="0" aria-label="Simulation time"><output id="stamp"></output></div><small>Amplitude and color scales stay fixed throughout each movie. Frames are actual simulated observations; no temporal interpolation. The surface shows the same two-dimensional displacement field as height, not a three-dimensional PDE. This is the first validation case, rather than a cohort summary.</small><script>const data=__DATA__;let i=0,playing=true,timer;const el=id=>document.getElementById(id);function show(){const group=data[el('boundary').value];const images=group[el('view').value];el('movie').src=images[i];el('stamp').textContent='t = '+group.times[i].toFixed(2);el('time').value=i;el('time').max=images.length-1;el('play').textContent=playing?'Pause':'Play'}function schedule(){clearTimeout(timer);if(playing)timer=setTimeout(()=>{i=(i+1)%data[el('boundary').value].times.length;show();schedule()},(i===0?400:i===data[el('boundary').value].times.length-1?900:100)/Number(el('speed').value))}el('play').onclick=()=>{playing=!playing;show();schedule()};el('restart').onclick=()=>{i=0;show();schedule()};el('time').oninput=()=>{i=Number(el('time').value);playing=false;show();schedule()};for(const name of ['boundary','view','speed'])el(name).onchange=()=>{show();schedule()};show();schedule();</script></body></html>'''


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--runs', required=True, type=Path)
    p.add_argument('--movie-job', default='movie01')
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    movie_cluster = args.runs/args.movie_job/'cluster'
    source = movie_cluster/'out/movie'
    exported = json.loads((source/'result.json').read_text())
    assert not exported['final_test_opened'] and not exported['training_performed'] and not exported['rom_recomputed']
    movie_hashes = checked_inputs(movie_cluster, [source/'result.json', source/'reflective.npz', source/'absorbing.npz'])
    records, webpage = {}, {}
    for boundary, label, bc in [('reflective', 'reflective01', 'dirichlet'), ('absorbing', 'absorbing02', 'absorbing')]:
        cluster = args.runs/label/'cluster'
        campaign, case = cluster/'out/campaign', exported['case']
        directory = campaign/bc
        run = json.loads((campaign/'result.json').read_text())
        seed = run['config']['optimizer_seeds'][0]
        arm = next(a for a in run['boundary_results'][bc]['arms'] if a['name']=='mlp' and a['optimizer_seed']==seed)
        assert arm['rollout']['refinement_passed']
        dt = arm['rollout']['primary_dt']
        parent_files = [campaign/'result.json', directory/'data_manifest.json', directory/'truth_spotchecks.npz', directory/'bank_tables.npz', directory/f'mlp_{seed}/rollouts.npz']
        parent_hashes = checked_inputs(cluster, parent_files)
        manifest = json.loads((directory/'data_manifest.json').read_text())
        validation = manifest['splits']['validation']
        scales, e0 = np.asarray(validation['scales'][case]), validation['initial_energy'][case]
        with np.load(source/(boundary+'.npz')) as saved:
            truth_u, truth_v, times, mass = saved['u'], saved['v'], saved['times'], saved['mass'].ravel()
            np.testing.assert_array_equal(saved['parameters'], np.asarray(validation['parameters'][case]))
        np.testing.assert_array_equal(times, np.arange(len(times))*run['config']['observation_dt'])
        with np.load(directory/'truth_spotchecks.npz') as spots:
            indices = np.rint(spots['times']/run['config']['observation_dt']).astype(int)
            spot_parity = max(float(np.max(abs(truth_u[indices].reshape(len(indices), -1)-spots['u'][case]))), float(np.max(abs(truth_v[indices].reshape(len(indices), -1)-spots['v'][case]))))
        assert spot_parity<1e-10
        with np.load(directory/'bank_tables.npz') as bank:
            g = bank['g']
            np.testing.assert_array_equal(mass, bank['mass'])
        with np.load(directory/f'mlp_{seed}/rollouts.npz') as saved:
            prefix = f'dt{dt}_case{case}_'
            assert np.all(saved[prefix+'completed'])
            predicted_u = saved[prefix+'coefficients']@g.T
            predicted_v = saved[prefix+'physical_velocity_coefficients']@g.T
            errors = state_metrics(predicted_u-truth_u.reshape(predicted_u.shape), predicted_v-truth_v.reshape(predicted_v.shape), mass, run['config']['n'], bc=='dirichlet', validation['parameters'][case][5], scales, e0)
            parity = {k:float(np.max(abs(v-saved[prefix+k+'_error']))) for k,v in errors.items()}
        assert max(parity.values())<1e-10, parity
        predicted_u = predicted_u.reshape(truth_u.shape)
        row = dict(parent_job=run['provenance']['job_id'], parameters=validation['parameters'][case], parent_source_sha256=parent_hashes, reference_spot_maximum_absolute_discrepancy=spot_parity, full_trajectory_metric_discrepancy=parity, original_mlp_dt=dt, optimizer_seed=seed, case=case, times=times.tolist(), animations={})
        webpage[boundary] = dict(times=times.tolist())
        for view in ('surface', 'top'):
            row['animations'][view], webpage[boundary][view] = render(args.out, boundary, view, times, truth_u, predicted_u, run['config']['n'], run['config']['latent'], seed)
        records[boundary] = row
    html = args.out/'2026-09-07-wave-evolution.html'
    html.write_text(HTML.replace('__DATA__', json.dumps(webpage, separators=(',', ':'))))
    record = dict(reference_export_job=exported['provenance']['job_id'], reference_export_source=exported['provenance']['source_commit'], reference_export_sha256=movie_hashes, reference_export_coordinator_sha256=exported['coordinator_script_sha256'], renderer_sha256=sha(Path(__file__)), selection='Same first validation case and first optimizer repeat as static plots.', no_temporal_interpolation=True, fixed_amplitude_scales=True, final_test_opened=False, boundaries=records, interactive_html=html.name, interactive_html_sha256=sha(html))
    (args.out/'2026-09-07-wave-evolution.json').write_text(json.dumps(record, indent=2)+'\n')
    print('interactive viewer', html, flush=True)


if __name__=='__main__':
    main()
