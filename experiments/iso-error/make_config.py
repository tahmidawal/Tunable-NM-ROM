"""iso-error: the sweep config = vendored config-t256.json with only the fields named in PROTOCOL.md section 3 changed.
    python make_config.py   ->  config-sweep256.json
"""
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
WIDTHS = [32, 48, 64, 96, 128, 192, 256, 384, 512]
# frozen compile mode per width (PROTOCOL.md section 2): dev6 timed_arm for 96..384; width-384 mode elsewhere
PRIMARY_GRAPHS = {32: True, 48: True, 64: True, 96: True, 128: True, 192: False, 256: False, 384: True, 512: True}
DEV6 = {96, 128, 192, 256, 384}


def name(Rp, graphs):
    return f'R{Rp}_lin_M{4 * Rp}_lat64_g0p01_fast_chol_clip_lamcarry_pred2__eng' + ('_graphs' if graphs else '')


def main():
    base = json.loads((HERE / 'vendor/experiments/burgers2d-test/config-t256.json').read_text())
    dev = json.loads((HERE / 'vendor/ref/dev6-b256-summary.json').read_text())
    for Rp in DEV6:        # the frozen mode is read from the dev6 record, not typed
        knob = [k for k in dev['knobs'] if k.startswith(f'R{Rp}_lin_M{4 * Rp}_lat64_g0p01_fast_chol_clip_lamcarry_pred2__')]
        assert len(knob) == 1, (Rp, knob)
        assert dev['knobs'][knob[0]]['timed_arm'] == name(Rp, PRIMARY_GRAPHS[Rp]), (Rp, dev['knobs'][knob[0]]['timed_arm'])
    proto = base['arms'][0]
    assert proto['Rp'] == 384 and proto['M'] == 1536
    cfg = copy.deepcopy(base)
    cfg['attempt'] = 'sweep256'
    cfg['ladder'] = WIDTHS
    arms = []
    for Rp in WIDTHS:
        for graphs in (PRIMARY_GRAPHS[Rp], not PRIMARY_GRAPHS[Rp]):
            s = copy.deepcopy(proto)
            s.update(Rp=Rp, M=4 * Rp, graphs=graphs, role='primary-mode' if graphs == PRIMARY_GRAPHS[Rp] else 'other-mode',
                     primary=bool(graphs == PRIMARY_GRAPHS[Rp]))
            arms.append(s)
    cfg['arms'] = arms
    cfg['audit_arms'] = ['fft_tight', name(384, True)]
    cfg['test_arms'] = dict(primary={str(Rp): name(Rp, PRIMARY_GRAPHS[Rp]) for Rp in WIDTHS}, primary_impl='eng',
                            secondary=[name(Rp, not PRIMARY_GRAPHS[Rp]) for Rp in WIDTHS])
    cfg['save_restricted'] = False
    cfg['recon_reps'] = 2
    cfg['purpose'] = 'iso-error PROTOCOL.md: span-width sweep of the frozen linear rung on the 64 test cases'
    changed = sorted(k for k in cfg if cfg[k] != base.get(k))
    assert changed == ['arms', 'attempt', 'audit_arms', 'ladder', 'purpose', 'recon_reps', 'save_restricted', 'test_arms'], changed
    (HERE / 'config-sweep256.json').write_text(json.dumps(cfg, indent=1) + '\n')
    print('arms', [name(s['Rp'], s['graphs']) for s in arms])


if __name__ == '__main__':
    main()
