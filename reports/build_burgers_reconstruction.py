"""Build the same-sample Burgers reconstruction report from immutable exports.

This consumes completed validation exports only. It never selects settings, fits
models, or reads sealed cases. Casewise failures remain in every denominator.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
from statistics import median
import tempfile


GATE = {"median_ratio_max": .8, "resolved_win_fraction_min": .75}
POD_ABSOLUTE_GAP_THRESHOLD = 1e-5
COHORT = {2: ("val32", 32, {32, 64, 128}, 256), 3: ("val64", 64, {16, 32}, 64)}
CONFIG_COUNTS = {2: {32: 77, 64: 77, 128: 145}, 3: {16: 93, 32: 93}}
LABEL = {
    "ordinary": "Selected classical reconstruction",
    "bank_base": "Fine-trained decoder",
    "bank_coarse": "Coarse-trained decoder",
    "pod_base": "Fine-data POD",
    "pod_coarse": "Coarse-data POD",
    "bank": "Learned decoder",
    "pod_matched": "POD, matched observations",
    "pod_stronger": "POD, extra fine samples",
    "bank_coarse_fixed": "Coarse-data decoder (fixed control)",
    "pod_coarse_fixed": "Coarse-data POD (fixed control)",
}
CLASSICAL = {"linear": "Linear", "cubic_notaknot": "Cubic", "sine_full": "Full sine"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(x):
    return isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def timing_median(values):
    return median(values) if values and all(finite(v) and v >= 0 for v in values) else None


def json_safe(value):
    """Retain failed/unavailable measurements without non-standard JSON numbers."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [json_safe(v) for v in value]
    return value


def normalize(path, expected_dimension):
    data = json.loads(Path(path).read_text())
    require(data["phase"] == "val", "Report accepts validation exports only")
    dim = data["dimension"]
    require(dim == expected_dimension, "Export dimension does not match its CLI argument")
    cohort, count, meshes, fine_mesh = COHORT[dim]
    rows = []
    for r in data["rows"] if dim == 2 else data["cases"]:
        if dim == 2:
            times = r["timings"]["host_query_seconds"]
            row = dict(dimension=dim, Lc=r["Lc"], Lf=r["Lf"], category=r["group"],
                       error=r["max_error"], eps=r["epsilon_reference"],
                       amplification=r["amplification"], strict=r["strict_coarse_control"],
                       seconds=timing_median(times),
                       fit_seconds=timing_median(r["timings"]["fit_seconds"]),
                       decode_seconds=timing_median(r["timings"]["decode_seconds"]))
        else:
            row = dict(dimension=dim, Lc=r["coarse_intervals"], Lf=r["fine_intervals"],
                       category=r["descriptor"]["category"], error=r["error"],
                       eps=r["reference_eps"], amplification=r["physical_amplification"],
                       strict=False, seconds=(r["timing_median"] or {}).get("host_query_seconds"),
                       fit_seconds=(r["timing_median"] or {}).get("fit_seconds"),
                       decode_seconds=(r["timing_median"] or {}).get("decode_seconds"))
        row.update({k: r[k] for k in ("source", "cohort", "case", "config", "status", "selected", "input_eligible")})
        for key in ('seconds','fit_seconds','decode_seconds'):
            if not finite(row[key]) or row[key] < 0:
                row[key] = None
        row["finite"] = row["status"] == "complete" and finite(row["error"])
        if not finite(row['error']):
            row['error'] = None
        amp = row['amplification']
        row['amplification_status'] = ('unavailable' if amp is None else
                                       'available' if finite(amp) and amp >= 0 else 'nonfinite_or_invalid')
        if row['amplification_status'] != 'available':
            row['amplification'] = None
        if dim == 2:
            alts = r['reference_level_sensitivity'] if row['source'] == 'A' else r['coarse_time_sensitivity']
            expected = 2 if row['source'] == 'A' else 1
            fields = ['output_change_vs_primary', 'error_vector_change' if row['source'] == 'A' else 'error_against_fixed_reference']
            sensitive = len(alts) == expected and all(
                len(a.get(k, [])) == 5 and all(finite(v) and v >= 0 for v in a[k])
                for a in alts for k in fields)
            row['prediction_change'] = max(v for a in alts for v in a['output_change_vs_primary']) if sensitive else None
            row['error_vector_change'] = max(v for a in alts for v in a['error_vector_change']) if sensitive and row['source'] == 'A' else None
            row['previous_error'] = max(alts[0]['error_against_fixed_reference']) if sensitive and row['source'] == 'B' else None
        else:
            alts = r['alternative_input_errors']
            sensitive = len(alts) == (2 if row['source'] == 'A' else 1) and all(finite(v) and v >= 0 for v in alts)
            row['prediction_change'] = r['input_sensitivity_max'] if sensitive else None
            row['error_vector_change'] = r['error_vector_sensitivity_max'] if sensitive and row['source'] == 'A' else None
            row['previous_error'] = alts[0] if sensitive and row['source'] == 'B' else None
        sensitive = (sensitive and finite(row['prediction_change']) and row['prediction_change'] >= 0
                     and (row['source'] == 'B' or (finite(row['error_vector_change']) and row['error_vector_change'] >= 0)))
        if not sensitive:
            row.update(prediction_change=None,error_vector_change=None,previous_error=None)
        row['sensitivity_complete'] = bool(sensitive)
        if row["status"] == "complete":
            error_times = r["error_t"] if dim == 2 else r["error_times"][1:]
            require(len(error_times) == 5 and all(finite(e) and e >= 0 for e in error_times), "Invalid evolved-time errors")
            require(row["finite"] and row["error"] >= 0, "Complete case has invalid error")
            require(math.isclose(row["error"], max(error_times), rel_tol=2e-14, abs_tol=1e-16), "Case error differs from time maximum")
        require(finite(row["eps"]) and row["eps"] >= 0, "Invalid reference uncertainty")
        require(type(row["case"]) is int, "Non-integer case identity")
        require(row["source"] in ("A", "B") and row["cohort"] == cohort, "Unexpected source or cohort")
        require(row["Lc"] in meshes and row["Lf"] == fine_mesh, "Unexpected mesh")
        rows.append(row)
    require(rows and all(r["dimension"] == dim for r in rows), "Empty or mixed dimension export")
    keys = [(r["Lc"], r["source"], r["case"], r["config"]) for r in rows]
    require(len(keys) == len(set(keys)), "Duplicate case/configuration entries")
    cohorts = {r["cohort"] for r in rows}
    require(len(cohorts) == 1, "Mixed cohorts")
    universe = set(range(count))
    require({r["case"] for r in rows} == universe, "Incomplete declared case identities")
    require({(r["Lc"], r["source"]) for r in rows} == {(lc, s) for lc in meshes for s in ("A", "B")}, "Missing mesh/source population")
    by_config = defaultdict(list)
    for row in rows:
        by_config[row["Lc"], row["source"], row["config"]].append(row)
    for rr in by_config.values():
        require({r["case"] for r in rr} == universe, "Incomplete declared cohort coverage")
        require(len({r["selected"] for r in rr}) == 1, "Selection differs across cases")
        require(len({r["category"] for r in rr}) == 1, "Category differs across cases")
    for lc in meshes:
        require(len({r["config"] for r in rows if r["Lc"] == lc and r["source"] == "A"}) == CONFIG_COUNTS[dim][lc], "Missing declared configurations")
        require({r["config"] for r in rows if r["Lc"] == lc and r["source"] == "A"} == {
            r["config"] for r in rows if r["Lc"] == lc and r["source"] == "B"}, "A/B configuration sets differ")
    return data, rows


def summarize(rows):
    valid = [r for r in rows if r["finite"]]
    errors = [r["error"] for r in valid]
    timings = {key: [r[key] for r in valid if finite(r[key])]
               for key in ("seconds", "fit_seconds", "decode_seconds")}
    sensitivity = [r for r in valid if r['sensitivity_complete']]
    invalid_amplification = sum(r['amplification_status'] == 'nonfinite_or_invalid' for r in rows)
    return dict(cases=len(rows), finite_cases=len(valid), failures=len(rows)-len(valid),
                input_admitted=sum(r["input_eligible"] for r in rows),
                median=median(errors) if errors else None, worst=max(errors) if errors else None,
                above_one_percent=sum(e > .01 for e in errors),
                **{key: median(values) if len(values) == len(rows) else None
                   for key, values in timings.items()},
                sensitivity_cases=len(sensitivity),
                prediction_change=max(r['prediction_change'] for r in sensitivity) if len(sensitivity) == len(rows) else None,
                error_vector_change=max(r['error_vector_change'] for r in sensitivity) if len(sensitivity) == len(rows) and rows[0]['source'] == 'A' else None,
                eps_max=max(r["eps"] for r in rows),
                invalid_amplification_cases=invalid_amplification,
                amplification=max((r["amplification"] for r in valid
                                   if finite(r["amplification"])), default=None) if not invalid_amplification else None)


def compare(method, baseline):
    a = {r["case"]: r for r in method}
    b = {r["case"]: r for r in baseline}
    require(a.keys() == b.keys() and a, "Comparison populations differ or are empty")
    sources = {r["source"] for r in method+baseline}
    require(len(sources) == 1, "Comparison mixes input sources")
    source = next(iter(sources))
    pairs = []
    for case in sorted(a):
        x, y = a[case], b[case]
        require(x["eps"] == y["eps"], "Methods must share reference qualification")
        ok = x["finite"] and y["finite"]
        gap = y["error"]-x["error"] if ok else None
        width = 1e-14 if source == "A" else 2*x["eps"]
        old_gap = (y['previous_error']-x['previous_error'] if source == 'B' and ok
                   and finite(y['previous_error']) and finite(x['previous_error']) else None)
        pairs.append(dict(case=case, finite=ok, win=bool(ok and gap > width),
                          loss=bool(ok and gap < -width), difference=gap,
                          time_classification_stable=((old_gap > width, old_gap < -width) == (gap > width, gap < -width)) if old_gap is not None else None,
                          input_admitted=x["input_eligible"] and y["input_eligible"]))
    full = all(p["finite"] for p in pairs)
    sm, sb = summarize(method), summarize(baseline)
    ratio = sm["median"]/sb["median"] if full and sb["median"] > 0 else None
    wins = sum(p["win"] for p in pairs)
    losses = sum(p["loss"] for p in pairs)
    method_worst_lower = max(max(0., r["error"]-r["eps"]) for r in method) if full and source == 'B' else None
    baseline_worst_upper = max(r["error"]+r["eps"] for r in baseline) if full and source == 'B' else None
    worst_ok = (method_worst_lower <= baseline_worst_upper) if full and source == 'B' else None
    if source != 'B':
        gate, gate_status, gate_reason = None, 'not_applicable', 'Source A is fixed-array reconstruction'
    elif sb['failures']:
        gate, gate_status, gate_reason = None, 'not_evaluable', 'Classical comparator has failed cases'
    elif sm['failures']:
        gate, gate_status, gate_reason = False, 'not_pass', 'Method has failed cases; complete accuracy comparison unavailable'
    elif ratio is None:
        gate, gate_status, gate_reason = None, 'not_evaluable', 'Baseline median is zero; ratio unavailable'
    else:
        gate = bool(ratio <= GATE['median_ratio_max'] and wins/len(pairs) >= GATE['resolved_win_fraction_min'] and worst_ok)
        gate_status, gate_reason = ('pass' if gate else 'not_pass'), 'Prespecified complete-cohort thresholds'
    return dict(method=method[0]["config"], baseline=baseline[0]["config"],
                source=source, cases=len(pairs), complete_pairs=sum(p["finite"] for p in pairs),
                reference_eps_max=sm['eps_max'],
                wins=wins, losses=losses, ties=sum(p["finite"] for p in pairs)-wins-losses,
                failed_pairs=sum(not p["finite"] for p in pairs), ratio_of_medians=ratio,
                time_pairs_checked=sum(p['time_classification_stable'] is not None for p in pairs),
                time_pairs_stable=sum(p['time_classification_stable'] is True for p in pairs),
                no_resolved_worst_deterioration=worst_ok if source == "B" else None,
                method_worst_lower=method_worst_lower, baseline_worst_upper=baseline_worst_upper,
                accuracy_gate=gate, gate_status=gate_status, gate_reason=gate_reason,
                input_admitted=sum(p["input_admitted"] for p in pairs), pairs=pairs)


def fmt(value, scale=1):
    return "—" if value is None else f"{value*scale:.4g}"


def table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |",
                      "| " + " | ".join("---" for _ in header) + " |"]
                     + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def diagnostic_evidence(roots, chosen):
    """Read immutable setup/oracle records without recomputing or refitting fields."""
    consumed, cache = {}, {}
    def read(path):
        if path not in cache:
            cache[path] = json.loads(path.read_text())
            consumed[str(path.resolve())] = sha(path)
        return cache[path]
    conditioning, setup_rows, oracle_rows = [], [], []
    for dim, root in roots.items():
        cohort, count, meshes, _ = COHORT[dim]
        plans_by_mesh, failures_by_mesh, declared_labels = {}, {}, {}
        for lc in sorted(meshes):
            path = root/f'plans/L{lc}/plans.json' if dim == 2 else root/f'L{lc}/plans/setup.json'
            setup = read(path)
            plans_by_mesh[lc] = ({r['name']:r for r in setup['plans']} if dim == 2 else setup['plans'])
            failures_by_mesh[lc] = ({r['name']:r for r in setup['failed_plans']} if dim == 2 else setup['failures'])
            declared_labels[lc] = {p['label'] for p in plans_by_mesh[lc].values() if 'label' in p}
            for name, failure in failures_by_mesh[lc].items():
                category = failure.get('group', failure.get('descriptor',{}).get('category'))
                if category != 'ordinary':
                    declared_labels[lc].add(name.rsplit('_',1)[0])
            setup_rows.append([dim, lc, fmt(setup['prepare_seconds'] if dim == 2 else setup['total_seconds']),
                               'Basis/factors; query compilation separate' if dim == 2 else 'Basis/factors and query warm-up/compilation'])
        for (rdim, lc, source, category), rows in sorted(chosen.items()):
            if rdim != dim or source != 'B' or category == 'ordinary':
                continue
            name = rows[0]['config']
            if name not in plans_by_mesh[lc]:
                failure = failures_by_mesh[lc][name]
                rank = failure.get('rank', failure.get('descriptor',{}).get('rank'))
                reason = str(failure.get('reason', failure.get('failure','unavailable'))).replace('|','/').replace('\n',' ')
                conditioning.append([dim, lc, LABEL[category], rank, '—', '—', '—', '—', 'Unavailable: '+reason])
                continue
            p = plans_by_mesh[lc][name]
            spectrum = p['singular_values']
            retained = p['sampled_effective_rank']
            condition = spectrum[0]/spectrum[retained-1] if retained and spectrum[retained-1] > 0 else None
            conditioning.append([dim, lc, LABEL[category], p['original_rank'], p['physical_rank'], retained,
                                 fmt(condition), fmt(p['physical_amplification']), 'Available'])
        # Primary-grid limits and the extra coarse-training-control grid. Other
        # meshes share the fine representation and remain in the raw archive.
        for lc in ([32, 128] if dim == 2 else [16]):
            per_label = {label: [] for label in declared_labels[lc]}
            for case in range(count):
                path = (root/f'oracle/L{lc}/{cohort}_c{case:03d}.json' if dim == 2
                        else root/f'L{lc}/{cohort}/c{case:03d}/oracle.json')
                for label, row in read(path).items():
                    require(label in per_label, 'Undeclared oracle representation')
                    values = row['error_t'] if dim == 2 else row['e_t'][1:]
                    if len(values) == 5 and all(finite(e) and e >= 0 for e in values):
                        per_label[label].append(max(values))
            for label, values in sorted(per_label.items()):
                full = len(values) == count
                oracle_rows.append([dim, lc, f'`{label}`', fmt(median(values) if full else None,100),
                                    fmt(max(values) if full else None,100), f'{len(values)}/{count}',
                                    'Available' if full else 'Unavailable or incomplete diagnostic'])
    return dict(conditioning=conditioning, setup=setup_rows, oracle=oracle_rows, consumed=consumed)


def pod_approximation_evidence(two_preparation, three_development, chosen):
    hashes, two_rows, three_rows = {}, [], []
    for family in ('base','coarse'):
        path=two_preparation/'bases'/f'pod_{family}.json'
        d=json.loads(path.read_text());hashes['2D_POD_'+family]=sha(path)
        require(finite(d['orthogonality_max']) and d['orthogonality_max']>=0,'Invalid POD orthogonality diagnostic')
        for rank in (128,256,384,512):
            residual=d['relative_eigen_residual'][:rank]
            require(len(residual)==rank and all(finite(x) and x>=0 for x in residual),'Invalid POD residual diagnostic')
            tail=d['projection_tail_fraction'][rank-1]
            require(finite(tail) and 0<=tail<=1,'Invalid POD training projection tail')
            two_rows.append([family,rank,fmt(max(residual)),fmt(tail),fmt(d['orthogonality_max'])])
    d=json.loads(three_development.read_text());hashes['3D_development_export']=sha(three_development)
    require(d['dimension']==3 and d['phase']=='dev','POD sensitivity must use the admitted 3D development export')
    require(len(d['cases'])==2544 and all(r['cohort']=='dev4' for r in d['cases']),'Wrong development population')
    for (dim,lc,source,category),rr in sorted(chosen.items()):
        if dim != 3 or not category.startswith('pod_'):
            continue
        config=rr[0]['config']
        cases=[r for r in d['cases'] if (r['coarse_intervals'],r['source'],r['config'])==(lc,source,config)]
        require(len(cases)==4 and {r['case'] for r in cases}==set(range(4)),'Incomplete POD development sensitivity population')
        require(all(r['selected'] for r in cases),'Validation POD setting differs from frozen development selection')
        good=[r for r in cases if r['approximate_pod_paired_times']==5
              and finite(r['approximate_pod_gap_max']) and r['approximate_pod_gap_max']>=0
              and isinstance(r['approximate_pod_gap_consequential'],bool)]
        for r in good:
            require(finite(r['reference_eps']) and r['reference_eps']>=0,'Invalid development reference uncertainty')
            expected=(r['approximate_pod_gap_max'] > 2*r['reference_eps'] or
                      r['approximate_pod_gap_max'] > POD_ABSOLUTE_GAP_THRESHOLD)
            require(r['approximate_pod_gap_consequential']==expected,'POD sensitivity flag differs from registered rule')
        three_rows.append([lc,source,LABEL[category],f'`{config}`',
                           fmt(max(r['approximate_pod_gap_max'] for r in good) if len(good)==len(cases) else None,100),
                           sum(r['approximate_pod_gap_consequential'] for r in good),f'{len(good)}/{len(cases)}'])
    return dict(hashes=hashes,absolute_gap_threshold=POD_ABSOLUTE_GAP_THRESHOLD,
                two_dimensional=two_rows,three_dimensional=three_rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--two-dimensional", type=Path, required=True)
    p.add_argument("--three-dimensional", type=Path, required=True)
    p.add_argument("--two-preparation", type=Path, required=True)
    p.add_argument("--three-development", type=Path, required=True)
    p.add_argument("--accepted-audit", type=Path, help="Independent result acceptance record bound to both exports and this generator")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    inputs = [args.two_dimensional, args.three_dimensional]
    allrows, provenance = [], []
    roots = {}
    for dim, path in zip((2, 3), inputs):
        data, rows = normalize(path, dim)
        roots[dim] = path.parent
        allrows.extend(rows)
        provenance.append(dict(path=str(path.resolve()), sha256=sha(path),
                               dimension=data["dimension"], phase=data["phase"],
                               runtime=data.get("runtime", data.get("execution"))))
    acceptance = None
    if args.accepted_audit:
        acceptance = json.loads(args.accepted_audit.read_text())
        require(acceptance.get('status') == 'accepted_for_screen', 'Result audit does not accept this screen')
        require(acceptance.get('export_sha256') == {str(p['dimension']):p['sha256'] for p in provenance}, 'Audit/export hashes differ')
        require(acceptance.get('generator_sha256') == sha(__file__), 'Audit/generator hash differs')
    groups = defaultdict(list)
    for r in allrows:
        groups[r["dimension"], r["Lc"], r["source"], r["config"]].append(r)
    chosen = {}
    for (dim, lc, source, config), rr in groups.items():
        if rr[0]["selected"]:
            key = dim, lc, source, rr[0]["category"]
            require(key not in chosen, "Multiple selected settings in one category")
            chosen[key] = rr
    unavailable_choices = sorted({(dim,lc,source,rr[0]['category']) for (dim,lc,source,_),rr in groups.items()} - chosen.keys())
    comparisons = []
    for (dim, lc, source, category), rr in sorted(chosen.items()):
        ordinary = chosen.get((dim, lc, source, "ordinary"))
        if category != "ordinary" and ordinary:
            comparisons.append(dict(dimension=dim, Lc=lc, category=category,
                                    **compare(rr, ordinary)))
    primary = [c for c in comparisons if c["source"] == "B"
               and c["Lc"] == {2: 32, 3: 16}[c["dimension"]]
               and c["category"] in ("bank_base", "bank")]
    require(len(primary) == len({c['dimension'] for c in primary}), 'Duplicate primary comparison')
    for c in comparisons:
        if c not in primary:
            c.update(accuracy_gate=None,gate_status='not_applicable',gate_reason='Not a prespecified primary comparison')
    primary_by_dim = {c["dimension"]: c for c in primary}
    for dim,lc in ((2,32),(3,16)):
        if dim not in primary_by_dim:
            fallback = groups[dim,lc,'B','linear']
            c = dict(dimension=dim,Lc=lc,source='B',category='bank_base' if dim == 2 else 'bank',
                     accuracy_gate=None,gate_status='not_evaluable',gate_reason='Required frozen bank or classical selection is unavailable',
                     cases=COHORT[dim][1],input_admitted=sum(r['input_eligible'] for r in fallback))
            primary.append(c);primary_by_dim[dim]=c
    fixed_maps = []
    for (dim,lc,source,category),rr in sorted(chosen.items()):
        if source == 'B':
            sampled = groups[dim,lc,'A',rr[0]['config']]
            fixed_maps.append(dict(dimension=dim,Lc=lc,category=category,config=rr[0]['config'],
                                   sampled_reference=summarize(sampled),actual_coarse=summarize(rr)))
    control_comparisons = []
    learned = [(dim,lc,source,category,rr,False) for (dim,lc,source,category),rr in sorted(chosen.items())
               if category.startswith('bank')]
    learned += [(dim,lc,source,rr[0]['category'],rr,True) for (dim,lc,source,_),rr in sorted(groups.items())
                if rr[0]['strict'] and rr[0]['category'].startswith('bank')]
    for dim,lc,source,category,rr,strict in learned:
        baselines = [(label,groups[dim,lc,source,name]) for name,label in CLASSICAL.items()]
        if strict:
            baselines.append(('POD fixed coarse-data control',groups[dim,lc,source,'pod_coarse_r384_ridge0']))
        else:
            pod_categories = ['pod_'+category.split('_',1)[1]] if dim == 2 else ['pod_matched','pod_stronger']
            baselines += [(LABEL[cat],chosen[dim,lc,source,cat]) for cat in pod_categories if (dim,lc,source,cat) in chosen]
        for label,baseline in baselines:
            result = compare(rr,baseline)
            result.update(accuracy_gate=None,gate_status='not_applicable',gate_reason='Descriptive control comparison')
            control_comparisons.append(dict(dimension=dim,Lc=lc,category=category,strict=strict,
                                            baseline_label=label,**result))
    conditioning_choices = dict(chosen)
    for (dim,lc,source,_),rr in groups.items():
        if rr[0]['strict'] and source == 'B':
            conditioning_choices[dim,lc,source,rr[0]['category']+'_fixed'] = rr
    evidence = diagnostic_evidence(roots, conditioning_choices)
    pod_evidence=pod_approximation_evidence(args.two_preparation,args.three_development,chosen)
    if acceptance:
        require(acceptance.get('auxiliary_sha256')==pod_evidence['hashes'],'Audit/approximation evidence hashes differ')
    sentences = []
    for dim in sorted({r["dimension"] for r in allrows}):
        c = primary_by_dim.get(dim)
        if c['gate_status'] == 'not_evaluable':
            sentences.append(f"The {dim}D primary accuracy gate is not evaluable: {c['gate_reason']}.")
        else:
            verdict = "passes" if c["accuracy_gate"] else "does not pass"
            sentences.append(f"The {dim}D decoder {verdict} the prespecified accuracy gate "
                             f"against development-selected classical reconstruction.")
    overview = []
    for dim, c in sorted(primary_by_dim.items()):
        lc = c['Lc']
        rows_by_method = [chosen.get((dim,lc,'B','bank_base' if dim == 2 else 'bank'))]
        rows_by_method += [groups[dim,lc,'B',name] for name in CLASSICAL]
        rows_by_method += [chosen.get((dim,lc,'B','ordinary'))]
        cells = []
        for rr in rows_by_method:
            if not rr:
                cells.append('unavailable')
            else:
                s = summarize(rr)
                cells.append(fmt(s['median'],100)+(f" ({s['failures']} failed)" if s['failures'] else ''))
        overview.append([f"{dim}D, {lc} → {groups[dim,lc,'B','linear'][0]['Lf']}", *cells])
    lines = ["# Burgers reconstruction from the same coarse samples", "",
             "This report compares classical and learned-basis reconstruction using identical "
             "coarse field samples. "+('The numbers are final for this bounded screen after independent result review; numerical-reference and training-information qualifications remain explicit.'
                                       if acceptance else 'The numbers are provisional pending independent returned-result and reporting review.'), "",
             " ".join(sentences), "",
             "The table below gives median errors on the coarsest actual-solver input grid in each dimension. "
             "Each entry is a percentage of the initial-field norm, after taking each case's largest evolved-time error. "
             "The decoder and selected classical settings were frozen using development cases; the three "
             "interpolation columns are fixed methods. Any failed cases are marked beside the finite-case median.", "",
             table(["Dimension, coarse → fine intervals", "Decoder (%)", "Linear (%)", "Cubic (%)", "Full sine (%)", "Selected classical (%)"], overview),
             "", "## What is being compared", "",
             "The screen uses the repository's scalar viscous Burgers families and exposed research cohorts.", "",
             ("Unavailable frozen selections: "+'; '.join(f'{d}D L{lc} {s} {LABEL[cat]}' for d,lc,s,cat in unavailable_choices)+'.'
              if unavailable_choices else 'Every declared method family has an available frozen selection.'), "",
             "Source A supplies exact samples of the stored fine-reference array. Source B supplies "
             "an actual coarse numerical solution. Every method receives the same field at the same "
             "physical time, coordinates and zero boundary values. Reconstructed fields are not advanced "
             "through time or corrected with a fine PDE solve. Sealed and out-of-distribution cohorts remain closed.", "",
             "The coarse inputs use central differences and Crank–Nicolson stepping with backward-Euler "
             "startup. Historical bank training values use the inherited upwind/backward-Euler recipe. "
             "That numerical-prior mismatch is part of the inherited system tested here.", "",
             "The decoder and POD have training-derived information about plausible fields. "
             "Interpolation has grid geometry and boundary information. Finer decoder evaluation "
             "does not add observations: any improvement comes from the learned representation "
             "and its regularization. Direct fitting to a fine reference is an oracle diagnostic, "
             "excluded from the reconstruction comparison.", "",
             "The classical candidate pool contains exact interpolation and regularized sine reconstructions. "
             "A filtered method generally changes the supplied coarse values, so the selected classical "
             "comparator is not necessarily interpolation in the strict sense. Linear, cubic and full "
             "sine interpolation are reported separately.", "",
             "Errors use the volume-weighted fine-grid norm, divided by the fixed initial-reference "
             "norm. Each case is scored by its maximum error across the evolved observation times:", "",
             r"$$e_i=\max_{t>0}\frac{\|\widehat u_i(t)-u_{i,\mathrm{ref}}(t)\|_{h_f}}{\|u_{i,\mathrm{ref}}(0)\|_{h_f}}.$$", "",
             "## Actual coarse solutions: selected settings", ""]
    score_rows = []
    for (dim, lc, source, category), rr in sorted(chosen.items()):
        if source != "B":
            continue
        s = summarize(rr)
        score_rows.append([dim, f"{lc} → {rr[0]['Lf']}", LABEL[category],
                           f"`{rr[0]['config']}`", fmt(s["median"],100), fmt(s["worst"],100),
                           s["above_one_percent"], f"{s['failures']}/{s['cases']}",
                           f"{s['input_admitted']}/{s['cases']}", fmt(s["fit_seconds"],1000),
                           fmt(s["decode_seconds"],1000), fmt(s["seconds"],1000)])
    lines += [table(["Dimension", "Intervals per axis", "Method family", "Frozen configuration",
                     "Median error (%)", "Worst error (%)", "Above 1%", "Failed / all", "Input admitted / all",
                     "Fit (ms)", "Decode (ms)", "Warm query (ms)"], score_rows), "",
              "When any fit fails, displayed error summaries describe finite cases only; the failure "
              "column preserves the full denominator, and the full-cohort improvement ratio is unavailable. "
              "An unadmitted coarse input still permits a same-input accuracy comparison, but its "
              "error cannot be attributed cleanly to spatial discretization alone.", "",
              "## Paired comparisons with selected classical reconstruction", ""]
    lines += [f"![Primary actual-coarse paired errors]({args.out.stem}.png)", "",
              "Each point is one case. Colors use the conditional numerical-reference resolution rule; "
              "they do not indicate statistical significance. The diagonal marks equal error.", ""]
    comparison_rows = []
    for c in comparisons:
        comparison_rows.append([c["dimension"], c["Lc"], c["source"], LABEL[c["category"]],
                                fmt(c["ratio_of_medians"]), fmt(c['reference_eps_max'],100) if c['source'] == 'B' else '—', c["wins"], c["ties"], c["losses"],
                                f"{c['failed_pairs']}/{c['cases']}",
                                f"{c['time_pairs_stable']}/{c['time_pairs_checked']}/{c['cases']}" if c['source'] == 'B' else '—'])
    lines += [table(["Dimension", "Coarse intervals", "Source", "Method family", "Median error ratio",
                     "Max reference ε (%)", "Wins", "Ties", "Losses", "Failed pairs / all", "Time check stable / checked / all"], comparison_rows), "",
              "The ratio is method median error divided by the selected classical method's median error; smaller is better. "
              "It is not the median of paired ratios. Source A wins describe completion of fixed numerical "
              "arrays, with a numerical tie tolerance. Source B wins exceed twice the stored reference "
              "uncertainty estimate. That estimate is a conditional numerical resolution rule, not a "
              "certified physical-error bound or a statistical confidence interval.", "",
              "The time check repeats each frozen reconstruction map on the preceding coarse-solver "
              "time-step level and asks whether its paired win/tie/loss classification changes. "
              "Missing diagnostics do not count as stable checks.", "",
              "## Holding the reconstruction map fixed", "",
              table(["Dimension", "Coarse intervals", "B-selected family", "Frozen configuration",
                     "A median error (%)", "B median error (%)", "A worst error (%)", "B worst error (%)",
                     "A / B failed cases"],
                    [[r['dimension'],r['Lc'],LABEL[r['category']],f"`{r['config']}`",
                      fmt(r['sampled_reference']['median'],100),fmt(r['actual_coarse']['median'],100),
                      fmt(r['sampled_reference']['worst'],100),fmt(r['actual_coarse']['worst'],100),
                      f"{r['sampled_reference']['failures']} / {r['actual_coarse']['failures']}"] for r in fixed_maps]), "",
              "This table applies the same B-selected map to both input sources. It separates the "
              "effect of changing observations from changing reconstruction settings. For the fixed "
              "linear maps tested here, the error vectors obey:", "",
              r"$$\widehat u_B-u_{\mathrm{ref}}=(\widehat u_A-u_{\mathrm{ref}})+T(y_B-y_A).$$", "",
              "Their norms do not form an additive error split. A is completion of the stored numerical "
              "array; B additionally contains coarse-solver input error, including any unresolved temporal contribution.", "",
              "## Decoder comparisons with literal interpolation and POD", "",
              table(["Dimension", "Coarse intervals", "Source", "Decoder configuration", "Comparator",
                     "Median error ratio", "Wins", "Ties", "Losses", "Failed pairs / all"],
                    [[c['dimension'],c['Lc'],c['source'],f"`{c['method']}`"+(' (fixed control)' if c['strict'] else ''),
                      c['baseline_label'],fmt(c['ratio_of_medians']),c['wins'],c['ties'],c['losses'],
                      f"{c['failed_pairs']}/{c['cases']}"] for c in control_comparisons]), "",
              "These comparisons use each fixed interpolation method and the separately frozen POD "
              "settings. They do not choose a new comparator from validation results. The fixed "
              "coarse-data decoder is paired with the fixed coarse-data POD control as well.", "",
              "## Fixed ordinary methods and coarse-training control", ""]
    controls = []
    for (dim, lc, source, config), rr in sorted(groups.items()):
        if config not in CLASSICAL and not rr[0]["strict"]:
            continue
        s = summarize(rr)
        label = CLASSICAL.get(config, LABEL[rr[0]["category"]] + " (fixed control)")
        controls.append([dim, lc, source, label, f"`{config}`", fmt(s["median"],100),
                         fmt(s["worst"],100), f"{s['failures']}/{s['cases']}"])
    lines += [table(["Dimension", "Coarse intervals", "Source", "Method", "Configuration",
                     "Median error (%)", "Worst error (%)", "Failed / all"], controls), "",
              "The coarse-training control uses its prespecified rank and no fitted ridge penalty. "
              "Other selected configurations use exposed development references; their selection "
              "therefore has fine-reference information. Both development and validation are exposed "
              "research cohorts, rather than sealed generalization tests. Coarse-data training refers "
              "to the solution values used in training; it does not erase the architecture's and "
              "recipe's earlier development history.", "",
              "## Sensitivity to the numerical inputs", "",
              table(["Dimension", "Coarse intervals", "Source", "Selected family", "Max output change (%)",
                     "Max A error-vector change (%)", "Complete / all"],
                    [[dim, lc, source, LABEL[category], fmt(summarize(rr)['prediction_change'],100),
                      fmt(summarize(rr)['error_vector_change'],100),
                      f"{summarize(rr)['sensitivity_cases']}/{len(rr)}"]
                     for (dim,lc,source,category),rr in sorted(chosen.items())]), "",
              "Source A varies both sampled inputs and targets across the saved reference levels; "
              "the error-vector change accounts for their dependence. Source B varies the coarse "
              "time-step level while keeping the fine target fixed. These are measured perturbations, "
              "not certified bounds on unknown physical error.", "",
              "## Do the samples determine the representation?", "",
              table(["Dimension", "Coarse intervals", "Selected B family", "Requested directions",
                     "Retained fine directions", "Sampled rank", "Retained-spectrum condition", "Amplification", "Status"],
                    evidence['conditioning']), "",
              "The fine-grid Gram cutoff can remove nearly dependent directions before fitting. "
              "A sampled rank below the retained fine rank leaves directions unidentified by the observations. "
              "Full sampled rank alone does not imply stability: the condition and the regularized map's "
              "physical amplification quantify sensitivity at different stages.", "",
              "## Fine-reference oracle limits — diagnostics only", "",
              table(["Dimension", "Coarse-grid plan", "Fine representation", "Oracle median error (%)",
                     "Oracle worst error (%)", "Complete / all", "Status"], evidence['oracle']), "",
              "These projections use the full fine target and are inadmissible as reconstruction methods. "
              "They diagnose whether a representation can express the target at all. A large oracle floor "
              "limits any coefficient fit; a small floor with poor coarse reconstruction points toward "
              "identification, regularization or input error. They do not measure prediction skill.", "",
              "## Qualifications and cost", "",
              "POD uses recipe-reproduced training data and an approximate randomized factorization. "
              "Recipe, parameter-table and observation-coordinate agreement do not establish byte identity "
              "with archived training arrays. The matched-observation spatial extension and the stronger "
              "comparator with extra fine values are distinct. These comparisons do not by themselves "
              "isolate a causal advantage of the network architecture.", "",
              "POD is uncentered, without a separately added mean field. Configurations named "
              "`covridge` use diagonal training second moments; they do not use an exact full "
              "covariance model. The approximate basis can retain off-diagonal second moments.", "",
              table(["2D training data", "Prefix rank", "Max relative eigen residual", "Training projection tail", "Full-basis orthogonality max"],
                    pod_evidence['two_dimensional']), "",
              table(["3D coarse intervals", "Source", "POD family", "Frozen configuration", "Max q1/q2 change (%)",
                     "Consequential cases", "Complete / all dev cases"],pod_evidence['three_dimensional']), "",
              "The q1/q2 comparison uses the two predeclared approximation levels on development cases. "
              "A case is consequential when an evolved output change exceeds either twice its reference "
              f"uncertainty or the fixed normalized-field threshold {POD_ABSOLUTE_GAP_THRESHOLD:g}. Passing this check does not certify "
              "global POD optimality. The preparation's independent residual probes measure aggregate "
              "residual energy, without a probe-spectrum certificate.", "",
              "Warm query time is the median of per-case medians for independently fitted single fields, "
              "including common host transfers. It excludes reusable basis evaluation, factorization and "
              "compilation. Accuracy and timing come from the same calls, with raw repetitions retained. "
              "Compare costs only within the same job and GPU; cross-dimensional timings are not speedup comparisons.", "",
              table(["Dimension", "Coarse intervals", "Recorded library setup (s)", "Recorded scope"], evidence['setup']), "",
              "Library setup covers the entire tested configuration sweep in that validation job, rather "
              "than the cost of deploying one selected method. Fine-coordinate basis evaluation was cached "
              "during preparation. These measurements support steady repeated-query comparisons, not a "
              "claim about total cost for a single uncached reconstruction. Separate fit/decode medians "
              "need not sum to the median end-to-end time.", "",
              "No derivative-accuracy claim is made. Reference-level perturbations, coarse-time-step "
              "perturbations, coefficient observability, amplification and POD convergence diagnostics "
              "remain part of the archived evidence and independent acceptance audit.", "",
              "## Prespecified larger-campaign gate", "",
              f"The primary actual-coarse comparison requires a median-error ratio no larger than "
              f"{GATE['median_ratio_max']:g}, resolved wins in at least "
              f"{GATE['resolved_win_fraction_min']*100:g}% of all declared cases, and no reference-resolved "
              "deterioration of the cohort's worst error. Ties and failed pairs do not count as wins. "
              "This is an accuracy-screen gate; training-information, conditioning and input qualifications "
              "still constrain the scientific claim. Worst-error deterioration is resolved only when "
              "the method's lower uncertainty envelope exceeds the selected classical method's upper envelope:", "",
              r"$$\max_i\max(0,e_{i,\mathrm{method}}-\epsilon_i)>\max_i(e_{i,\mathrm{classical}}+\epsilon_i).$$", "",
              table(["Dimension", "Primary coarse intervals", "Accuracy gate", "Input admitted / all"],
                    [[dim, c["Lc"], {'pass':'Pass','not_pass':'Does not pass','not_evaluable':'Not evaluable'}[c['gate_status']],
                      f"{c['input_admitted']}/{c['cases']}"] for dim,c in sorted(primary_by_dim.items())]),
              "", "## Provenance", ""]
    for item in provenance:
        lines.append(f"- {item['dimension']}D validation export: `{item['path']}`; SHA-256 `{item['sha256']}`.")
    lines += ["", "The [protocol and controls](DESIGN.md) specify the physical times, meshes, training information, "
              "fitting rules and prospective selection criteria."]
    if acceptance:
        lines += ["", "The [independent acceptance record](ACCEPTED-SCREEN.json) binds the exports, report generator "
                  "and auxiliary POD evidence used for this publication."]
    lines += ["", "The companion JSON contains normalized selected-case records, full comparison pairs, "
              "runtime metadata and the gate constants used to generate these tables.", "",
              f"The [complete configuration table]({args.out.stem}.configurations.csv) retains every "
              "tested setting, including failed and unselected configurations. Error summaries in that "
              "table describe finite cases and must be read beside the failure count; blank timings "
              "mean that a complete timing measurement is unavailable.", "",
              "## Plain-language glossary", "",
              "- **Dimension / D:** number of spatial directions.",
              "- **Scalar / case:** one dependent field / one viscosity and initial-field choice.",
              "- **Central differences / upwind / Crank–Nicolson / backward Euler:** spatial difference "
              "rules using neighbors on both sides / a spatial rule following the flow direction / "
              "a time rule averaging old and new rates / an implicit time rule using the new rate.",
              "- **Intervals per axis / coarse intervals:** number of mesh cells along each direction; "
              "the full grid includes both boundary nodes.",
              "- **Source A / B:** sampled fine-reference array / actual coarse numerical solution.",
              "- **Decoder / learned basis:** coordinate-dependent functions learned from training fields; "
              "only their coefficients are fitted to the supplied coarse field.",
              "- **POD:** proper orthogonal decomposition, a linear basis extracted from training snapshots.",
              "- **Uncentered / second moments / covridge:** snapshots without mean subtraction / "
              "average coefficient products / a penalty using only the diagonal of those products.",
              "- **Relative eigen residual / projection tail / orthogonality max:** mismatch in the approximate "
              "training covariance eigen-equation / fraction of training energy left outside the basis / "
              "largest departure of the basis inner products from an identity matrix.",
              "- **q1/q2 / consequential cases:** two predeclared randomized-POD approximation levels / "
              "development cases where their reconstructed outputs differ beyond the registered threshold.",
              "- **Bank q2 identifier:** bookkeeping in the saved configuration name; the learned bank is frozen "
              "and has no POD power-iteration parameter.",
              "- **Frozen configuration:** rank, penalty and method chosen before validation was run.",
              "- **Rank / ridge:** requested basis prefix size before numerical pruning / penalty that discourages unstable coefficients.",
              "- **Requested directions / retained fine directions / sampled rank:** initial basis size / "
              "directions kept after removing numerical dependence / directions distinguishable on the coarse grid.",
              "- **Retained-spectrum condition:** largest divided by smallest retained singular value of the "
              "weighted observation matrix; larger values indicate a more sensitive coefficient fit.",
              "- **Gram cutoff / library setup:** numerical rule for removing dependent basis directions / "
              "preparing all candidate reconstruction maps before querying them.",
              "- **Median error / worst error / above 1%:** middle case error / largest finite case error / "
              "count of finite cases whose error exceeds the stated threshold.",
              "- **Failed / all:** unusable numerical fits and the full declared cohort size; failures are never silently dropped.",
              "- **Input admitted:** coarse computation passed its fixed solver and time-step checks.",
              "- **Warm query:** fitting and evaluating one field after reusable setup and GPU warm-up.",
              "- **Wins / ties / losses:** paired comparisons using the source-specific resolution rule; failed pairs are separate.",
              "- **Median error ratio:** method cohort median divided by the named comparator's cohort median.",
              "- **Classical reconstruction:** geometry-based interpolation or a fixed spectral filter; "
              "its chosen regularization can use development references, but it has no learned spatial bank.",
              "- **Full sine / cubic / linear:** boundary-compatible spectral interpolation / tensor cubic spline / tensor linear interpolation.",
              "- **Matched observations / extra fine samples:** training observations matching the bank's recorded sampling "
              "pattern / a deliberately stronger comparator with additional fine-grid values.",
              "- **Oracle:** a diagnostic fitted directly to fine targets, unavailable to an admissible reconstruction method.",
              "- **Observability / amplification:** whether coarse samples identify basis coefficients / how much an input "
              "perturbation can change the reconstructed field.",
              "- **Invalid amplification cases:** entries with non-finite or negative amplification; "
              "their configuration maximum is unavailable, rather than silently ignoring them.",
              "- **Reference uncertainty:** estimated numerical uncertainty of the fine comparison field.",
              "- **Max reference ε:** largest case-specific reference uncertainty in the displayed cohort; "
              "paired decisions and worst-error envelopes still use each case's own value.",
              "- **Time check stable / checked / all:** paired classifications unchanged under the preceding coarse "
              "time-step level / pairs with complete perturbation evidence / all declared pairs.",
              "- **Output change / error-vector change:** change in reconstructed fields / change in "
              "reconstruction-minus-reference fields when numerical inputs are perturbed.",
              "- **Exposed / sealed / out-of-distribution:** already available for method development / reserved and unopened / "
              "outside the tested training-family conditions.",
              "- **PDE / rollout:** partial differential equation / advancing a state through time.", ""]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json_content=json.dumps(json_safe(dict(
        provenance=provenance, acceptance=acceptance, gate=GATE, selected_cases=[r for r in allrows if r['selected'] or r['strict']], comparisons=comparisons,
        primary=primary, fixed_maps=fixed_maps, control_comparisons=control_comparisons,
        unavailable_choices=unavailable_choices,diagnostic_evidence=evidence,
        pod_approximation=pod_evidence)), indent=2, allow_nan=False)+"\n"
    configuration_rows = []
    for (dim, lc, source, config), rr in sorted(groups.items()):
        configuration_rows.append(dict(dimension=dim, coarse_intervals=lc, fine_intervals=rr[0]['Lf'],
                                       source=source, configuration=config, category=rr[0]['category'],
                                       selected=rr[0]['selected'], strict_control=rr[0]['strict'], **summarize(rr)))
    csv_buffer=io.StringIO(newline='')
    writer=csv.DictWriter(csv_buffer,fieldnames=list(configuration_rows[0]))
    writer.writeheader();writer.writerows(configuration_rows)
    # Serialize every artifact before touching published outputs. Install the
    # markdown last so it cannot announce companions that failed serialization.
    outputs={args.out.with_suffix('.json'):json_content,
             args.out.with_name(args.out.stem+'.configurations.csv'):csv_buffer.getvalue(),
             args.out:'\n'.join(lines)}
    with tempfile.TemporaryDirectory(prefix='.reconstruction-build-',dir=args.out.parent) as temporary:
        for target,content in outputs.items():
            Path(temporary,target.name).write_text(content)
        for target in outputs:
            os.replace(Path(temporary,target.name),target)
    print(args.out)


if __name__ == "__main__":
    main()
