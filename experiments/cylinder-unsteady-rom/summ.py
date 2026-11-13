"""Print a compact table from R4 JSON files (values read from the saved JSON)."""
import json, sys
for f in sys.argv[1:]:
    d = json.load(open(f)); print(f, 'case', d['case'], 'dt', d['dt'])
    for n, r in d['arms'].items():
        if r['status'] != 'completed':
            print(f'  {n:10s} FAILED {r.get("error")}'); continue
        t = r['targets']; met = sum(v['met'] for v in t.values())
        print(f"  {n:10s} dim {r['total_unknowns']:3d} met {met}/6 e_u rms {t['e_u_rms']['value']:.3e} max {t['e_u_max']['value']:.3e} "
              f"e_p {t['e_p_rms']['value']:.3e} St {t['St_rel']['value']:.2e} CD {t['CD_max_rel']['value']:.2e} CL {t['CL_max_abs']['value']:.2e} "
              f"div {r['divergence']['median']:.2e} floor {r['floor']['velocity_with_supremizers']['median']:.2e} roll {r['rollout_seconds']:.2f}s newton {r['newton_iterations']}")
