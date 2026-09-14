"""Generate the active-worker handoff from retained audits; append only with --write."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--write',action='store_true');a=p.parse_args()
    def read(name):return json.loads((HERE/'checks'/name).read_text())
    launch=read('collection-monitor-launch.json');ref=read('refinement02-reference-audit.json')
    diag=read('refinement02-diagnosis-audit.json');data=read('live-dataset-audit.json')
    selected=ref['gate']['by_output']['256']['selected'];rom=diag['summary']['rom'];fom=diag['summary']['same_nt1e-2_dt005']
    date=time.strftime('%Y-%m-%d',time.gmtime())
    current=f'**Burgers first-stage worker — reference and diagnosis audited; data generation active.** Job `{launch["job_id"]}` is running in its unique A100 allocation. The refined independent reference gate passes; the inherited ROM is slower and less accurate than an efficient same-job FOM. The bounded CPU collector (PID `{launch["pid"]}`) owns completion checks, full raw Git archiving and exact job-directory cleanup, with a cluster-generated dataset handoff cache for the FNO owner. Inspect `worktrees/2026-09-14-no-burgers/experiments/neural-operator-burgers/checks/refinement02-collection-status.json` for its actual phase; no collection completion or accuracy/speed win is claimed yet. The broader campaign remains open, final cohorts sealed, and no merge occurred.'
    text=f'\n\n## {date}\n### Burgers active-worker handoff — refined reference passes; current ROM does not beat efficient FOM\n\n'
    text+=f'The original failed reference calibration remains archived unchanged. The focused refinement now passes on all {ref["cases_checked"]} independent calibration cases: {selected["intervals"]} intervals, time step {selected["dt"]:.16g}, worst empirical margin {selected["worst_empirical_margin"]:.16g} against the unchanged {ref["gate"]["empirical_budget"]:.16g} budget. The full owner NumPy audit checked {ref["solves_checked"]} saved fields and their step records, source hashes, original cached source/protocol/index/field links, requested initial fields and recorded refinement arithmetic. This is empirical evidence, not a rigorous continuum certificate.\n\n'
    text+=f'The independent diagnosis audit checked {diag["invocations_checked"]} complete paired invocations on the same {diag["cases_checked"]} cases; maximum reproduced-metric discrepancy is {diag["maximum_absolute_metric_discrepancy"]:.16g}. '
    text+=f'Native ROM median GPU/complete-host times are {rom["gpu_median_ms"]:.9f}/{rom["host_to_host_median_ms"]:.9f} ms with worst candidate-relative error {100*rom["worst_fixed_initial_error"]:.9f}%; the native FOM at Newton tolerance 0.01, linear tolerance 0.5 and time step 0.005 takes {fom["gpu_median_ms"]:.9f}/{fom["host_to_host_median_ms"]:.9f} ms with {100*fom["worst_fixed_initial_error"]:.9f}% error. Both columns have {rom["failed_stopping_invocations"]}/{fom["failed_stopping_invocations"]} failed stopping checks. The ROM is slower and less accurate here; no advantage over efficient FOM or neural operators is established. Full repetition arrays, all additional FOM presets, projection/fit/evolved-field errors and sampled-state quadrature discrepancies remain in the diagnosis evidence.\n\n'
    text+='The original checkpoint has unmatched training history and is only a pilot diagnostic. The archive retains the native fitted initial output; future deployable comparisons must return supplied initial output exactly for all methods while still charging internal fitting/evolution and keeping compression error separate. The bank, best-found nonlinear fit and online errors are not an additive decomposition; the multistart fit is not globally certified. The prepared quadrature-ablation source has not been run on a GPU.\n\n'
    text+=f'The first {data["total_records_checked"]} generated training records independently passed schema, seed, sampled initial field, finite float64 boundary and every saved solver-residual check. Bulk generation remains active under the existing total lane GPU budget. Numerical execution source is `5169c0950e95eb666ada753f358032be2c5a9585`; collector source is `{launch["source_commit"]}`. '
    text+=f'Local CPU monitor PID `{launch["pid"]}` is bounded to {launch["deadline_hours"]} hours and submits no GPU job. It will checksum-collect the full raw archive, rerun reference/diagnosis/dataset audits, retain compressed chunks and manifests in Git, then remove only the exact completed job directory. A verified cluster-generated cache is planned at `{launch["handoff_cache"]}` for subsequent FNO copying; original indices and their calibration-path relocation are retained explicitly. Its current state and failure status are durable in the checks directory. On successful completion it updates only this owner’s current-state paragraph and appends a locked canonical closing entry. Training/operator comparisons, later diagnosis-driven changes and the merge decision remain open.\n'
    if a.write:
        spec=importlib.util.spec_from_file_location('collector',HERE/'cluster/collect_when_done.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.append(text,current=current)
    else:print(text)


if __name__=='__main__':main()
