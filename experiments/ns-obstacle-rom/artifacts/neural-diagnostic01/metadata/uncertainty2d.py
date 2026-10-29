"""A1 conditional reference intervals; no anchor transfer or uncertainty relaxation."""
from pathlib import Path
import json,math,re
from fe_adapter import sha_file
from qualification import require_qualification
from pilot2d import DEV_RE,VALIDATION_RE

METRICS=('velocity_relative','pressure_scaled','drag_relative','transverse_absolute')
LIMITS=dict(zip(METRICS,(.01/5,.02/5,.02/5,.002/5)))
RULE_ALGORITHM='sha256_sorted_named_c_arrays_shape_dtype_v1'


def finite_nonnegative(value):
    if type(value) not in (int,float) or not math.isfinite(value) or value<0:raise ValueError('finite nonnegative uncertainty metric required')
    return float(value)


def equal(actual,expected):
    if not math.isclose(actual,expected,rel_tol=1e-12,abs_tol=0):raise ValueError('uncertainty arithmetic replay differs')


def digest(value):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):raise ValueError('explicit evidence digest required')
    return value


def metric_delta(row,name):
    raw=row['raw_norms'];names=('h_fine','p_same_geometry','p_combined','solver','quadrature')
    if set(raw)!=set(names) or row.get('p_definition')!='max(p_same_geometry,p_combined)':raise ValueError('both fixed P4-v-P3 contrasts required')
    values={}
    for key in names:
        record=raw[key];numerator=finite_nonnegative(record['numerator']);denominator=finite_nonnegative(record['denominator'])
        if denominator<=0:raise ValueError('positive raw metric denominator required')
        value=finite_nonnegative(record['value']);equal(value,numerator/denominator);values[key]=value
    for key in ('h_fine','solver','quadrature'):equal(finite_nonnegative(row[key]),values[key])
    equal(finite_nonnegative(row['p']),max(values['p_same_geometry'],values['p_combined']))
    delta=2*max(row['h_fine'],row['p'])+row['solver']+row['quadrature'];equal(finite_nonnegative(row['delta']),delta)
    if delta>LIMITS[name]:raise ValueError('reference uncertainty exceeds frozen one-fifth comparison budget')
    trend=row['resolved_h']
    for key in ('earlier','finest'):
        interval=trend[key];d=finite_nonnegative(interval['difference']);noise=finite_nonnegative(interval['pair_noise'])
        expected=[max(0.,d-noise),d+noise]
        if not isinstance(interval['interval'],list) or len(interval['interval'])!=2:raise ValueError('explicit h/noise interval required')
        for actual,value in zip(interval['interval'],expected):equal(finite_nonnegative(actual),value)
        if interval.get('resolved') is not True or not d>noise:raise ValueError('unresolved h difference')
    equal(trend['finest']['difference'],row['h_fine'])
    if trend.get('decreasing') is not True or not trend['earlier']['interval'][0]>trend['finest']['interval'][1]:
        raise ValueError('noise-separated decreasing h trend required')
    evidence=row['evidence_sha256']
    if not isinstance(evidence,dict) or not evidence:raise ValueError('complete constituent evidence bindings required')
    for value in evidence.values():digest(value)
    return delta


def load(path,expected_sha,review_path,review_sha,qualification,truth_sha,rule_sha,split):
    if sha_file(path)!=expected_sha or sha_file(review_path)!=review_sha:raise ValueError('released uncertainty/review differs')
    record=json.loads(Path(path).read_text());review=json.loads(Path(review_path).read_text())
    if (record.get('schema')!='ns-obstacle-reference-conditional-uncertainty-v1' or record.get('split')!=split or
        review.get('schema')!='ns-obstacle-reference-conditional-uncertainty-review-v1' or review.get('passed') is not True or
        review.get('uncertainty_sha256')!=expected_sha or review.get('reference_admission_sha256')!=truth_sha):
        raise ValueError('independently reviewed per-population uncertainty required')
    for item in (record,review):
        if require_qualification(item.get('reference_qualification'))!=require_qualification(qualification):raise ValueError('uncertainty scope differs')
    if record.get('reference_admission_sha256')!=truth_sha or record.get('volume_rule_sha256')!=rule_sha or record.get('volume_rule_hash_algorithm')!=RULE_ALGORITHM:
        raise ValueError('actual admitted truth/common rule identity differs')
    physical=record['physical_rule_identity']
    if set(physical)!={'points_sha256','weights_sha256','hash_algorithm','original_container_sha256'} or not isinstance(physical['hash_algorithm'],str) or not physical['hash_algorithm']:
        raise ValueError('separate physical semantic rule/container identities required')
    for key in ('points_sha256','weights_sha256','original_container_sha256'):digest(physical[key])
    expected={'development':DEV_RE,'validation':VALIDATION_RE}.get(split)
    if expected is None or [row['Re'] for row in record['rows']]!=list(expected):raise ValueError('explicit exact per-Re uncertainty; no anchor inheritance')
    result={}
    for row in record['rows']:
        if set(row['metrics'])!=set(METRICS):raise ValueError('all primary uncertainty metrics required')
        result[row['Re']]={key:metric_delta(row['metrics'][key],key) for key in METRICS}
    return result


def joint_match(bank_errors,competitor_errors,deltas):
    intervals={}
    for key in METRICS:
        bank=finite_nonnegative(bank_errors[key]);competitor=finite_nonnegative(competitor_errors[key]);delta=finite_nonnegative(deltas[key])
        lower=max(0.,bank-delta);upper=competitor+delta
        intervals[key]={'bank_error':bank,'competitor_error':competitor,'delta':delta,
            'bank_interval':[lower,bank+delta],'competitor_interval':[max(0.,competitor-delta),upper],
            'resolved_match':upper<=lower}
    return {'matched':all(row['resolved_match'] for row in intervals.values()),'primary_metrics':intervals,
        'conditional_not_rigorous':True,'criterion':'competitor upper <= bank lower for every primary output'}
