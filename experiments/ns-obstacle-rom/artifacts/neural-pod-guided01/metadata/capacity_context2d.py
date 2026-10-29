"""Temporary reviewed measured-count context. Defaults and all mathematical caps persist."""
from pathlib import Path
from contextlib import contextmanager
import hashlib,json
from qualification import require_qualification

MEASUREMENT_SHA='d0bee2f082f344dd4bc01f14b0ea985e245261333ea2537caf386c17ab1e5d2d'
FACTOR_SHA='85734edc5497d3bde956d4b71179875819e80e777f41dd8fe421a0b3f0e706cd'
CONTROL_REVIEW_SHA='8198c000885a5cd153c9317ec70d771e35357a5c5f5d1d9a526f4f9acc0992e3'
MESH_SHA='528ffff38bb0f4d6d542f1abd4cba7c9ca2ba0427871caca806527c63ae6b092'
LAYOUT_SHA='54d8cb2354693862d69c265c4aa0e3a9eac16a62518484ae8fef26f0a82f3f44'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def member(base,descriptor):
    path=(base/descriptor['path']).resolve()
    if sha(path)!=descriptor['sha256']:raise ValueError('capacity evidence changed')
    return path


def validate(release_path,release,admission,expected_source,caps,prepared_metadata=None):
    if 'capacity_amendment' not in release:return None
    base=Path(release_path).parent;amend_path=member(base,release['capacity_amendment']);review_path=member(base,release['capacity_review'])
    amend=json.loads(amend_path.read_text());review=json.loads(review_path.read_text())
    if (amend.get('schema')!='ns-obstacle-rom-count-capacity-amendment-v1' or
        amend.get('source_sha256')!=expected_source or amend.get('scientific_changes') is not False or
        review.get('schema')!='ns-obstacle-rom-count-capacity-review-v1' or review.get('passed') is not True or
        review.get('amendment_sha256')!=sha(amend_path) or review.get('source_sha256')!=expected_source):
        raise ValueError('independent exact-source resource-only amendment required')
    qualification=require_qualification(admission)
    if require_qualification(amend.get('reference_qualification'))!=qualification or amend.get('reference_admission_sha256')!=admission['reference_admission_sha256']:
        raise ValueError('capacity amendment differs from actual scoped data')
    if amend.get('original_caps')!=caps:raise ValueError('capacity defaults changed or context nested')
    expected={'measurement':MEASUREMENT_SHA,'factor_reuse':FACTOR_SHA,'native_control_review':CONTROL_REVIEW_SHA}
    evidence={}
    for key,digest in expected.items():
        path=member(amend_path.parent,amend[key])
        if sha(path)!=digest:raise ValueError('exact independently accepted native evidence required: '+key)
        evidence[key]=path
    measurement=json.loads(evidence['measurement'].read_text());factor=json.loads(evidence['factor_reuse'].read_text())
    if measurement.get('passed') is not True or factor.get('passed') is not True or measurement.get('layout_sha256')!=LAYOUT_SHA:
        raise ValueError('native measurement/reuse scope mismatch')
    if prepared_metadata is not None and prepared_metadata.get('layout_sha256')!=LAYOUT_SHA:
        raise ValueError('prepared layout differs from measured target')
    for case in admission['cases']:
        if case['files']['mesh.vol.bin']!=MESH_SHA:raise ValueError('admitted mesh differs from measured fine target')
        if prepared_metadata is None:
            layout_path=Path(case['resolved_directory'])/'fe_layout.json';layout=json.loads(layout_path.read_text())
            if sha(layout_path)!=case['files']['fe_layout.json'] or layout.get('sha256')!=LAYOUT_SHA:
                raise ValueError('admitted layout differs from measured fine target')
    counts=measurement['counts'];proposed=dict(caps)
    proposed.update(mixed_dofs=counts['velocity_dofs']+counts['pressure_dofs'],elements=counts['elements'],
        volume_points=counts['volume_points'],matrix_nnz=max(row['native_nnz'] for row in measurement['metrics'].values()))
    if amend.get('proposed_caps')!=proposed:raise ValueError('only exact measured count ceilings may change')
    if release.get('point_block_volume') is not True or release.get('verify_primary_volume') is not True:
        raise ValueError('fine capacity requires blocked export and complete fine primary replay')
    return {'caps':proposed,'amendment_sha256':sha(amend_path),'review_sha256':sha(review_path),
        'measurement_sha256':MEASUREMENT_SHA,'factor_reuse_sha256':FACTOR_SHA,'native_control_review_sha256':CONTROL_REVIEW_SHA,
        'layout_sha256':LAYOUT_SHA,'mesh_sha256':MESH_SHA,'scientific_changes':False}


@contextmanager
def approved_capacity(release_path,release,admission,expected_source,caps):
    proof=validate(release_path,release,admission,expected_source,caps);original=dict(caps)
    try:
        if proof is not None:caps.update(proof['caps'])
        yield proof
    finally:caps.clear();caps.update(original)


def consumer_caps(metadata,release_path,release,defaults):
    """Revalidate count provenance in fresh consumers; never mutate global CAPS."""
    if metadata.get('caps')==defaults and 'capacity_amendment' not in release:return dict(defaults)
    if metadata.get('volume_storage')!='point-block-v1':raise ValueError('fine prepared input must remain point-blocked')
    from prepare2d import preparation_digest
    from fe_adapter import adapter_digest
    qualification=require_qualification(metadata.get('reference_qualification'))
    admission=dict(qualification,reference_admission_sha256=metadata['admission_evidence']['reference_admission_sha256'],
                   cases=metadata['source_case_files'])
    if len(admission['cases'])!=9 or metadata.get('adapter_source_sha256')!=adapter_digest():
        raise ValueError('complete exact-source fine training preparation required')
    proof=validate(release_path,release,admission,preparation_digest(),defaults,metadata)
    if proof is None or metadata.get('caps')!=proof['caps']:raise ValueError('prepared count caps lack exact reviewed provenance')
    base=Path(release_path).parent;result_path=member(base,release['native_preparation_result'])
    review_path=member(base,release['native_preparation_review']);result=json.loads(result_path.read_text());review=json.loads(review_path.read_text())
    if (result.get('schema')!='ns-obstacle-native-preparation-result-v1' or result.get('source_sha256')!=preparation_digest() or
        result.get('status')!='execution_completed_pending_independent_review' or result.get('capacity_context')!=proof or
        result.get('adapter_manifest_sha256')!=release['adapter_manifest_sha256'] or
        result.get('reference_admission_sha256')!=admission['reference_admission_sha256'] or
        review.get('schema')!='ns-obstacle-native-preparation-review-v1' or review.get('passed') is not True or
        review.get('result_sha256')!=sha(result_path) or review.get('source_sha256')!=preparation_digest() or
        review.get('adapter_manifest_sha256')!=release['adapter_manifest_sha256'] or
        review.get('capacity_amendment_sha256')!=proof['amendment_sha256']):
        raise ValueError('independently reviewed actual fine preparation/context required')
    if require_qualification(result.get('reference_qualification'))!=qualification or require_qualification(review.get('reference_qualification'))!=qualification:
        raise ValueError('fine preparation review scope differs')
    return dict(proof['caps'])
