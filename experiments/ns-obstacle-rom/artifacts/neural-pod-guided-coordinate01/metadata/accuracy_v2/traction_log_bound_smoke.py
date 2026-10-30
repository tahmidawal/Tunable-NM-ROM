"""Conservative finite-scalar serialized sizes for actual trace optimizer row keys."""
import ast,json,hashlib
from pathlib import Path

def check():
    source=Path(__file__).with_name('traction_refine.py');helper=source.with_name('divergence_objective.py')
    tree=ast.parse(source.read_text());ht=ast.parse(helper.read_text())
    metrics_node=next(n for n in ast.walk(ht) if isinstance(n,ast.Return) and isinstance(n.value,ast.Tuple)).value.elts[1]
    metric_keys=[k.value for k in metrics_node.keys]
    traction=source.with_name('boundary_traction_objective.py');tt=ast.parse(traction.read_text())
    tn=next(n for n in ast.walk(tt) if isinstance(n,ast.Return) and isinstance(n.value,ast.Tuple)).value.elts[1]
    metric_keys += ['traction_'+k.value for k in tn.keys]
    metric_keys += [n.slice.value for n in ast.walk(tree) if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id=='parts_sum' and isinstance(n.slice,ast.Constant) and isinstance(n.ctx,ast.Store)]
    metric_keys=list(dict.fromkeys(metric_keys))
    # -maxfloat has the longest normal finite binary64 JSON rendering. Use it
    # even for integer fields, conservatively exceeding their frozen max widths.
    big=-1.7976931348623157e308
    metrics={k:([big]*4 if k.endswith('_by_mode') else big) for k in metric_keys};metrics['seconds']=big
    metrics['traction_integrated_traction_error_by_component_mode']=[[big]*4 for _ in range(2)]
    identity=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='identity' for t in n.targets))
    identity_keys=[k.value for k in identity.keys]
    bools={'accepted','curvature_kept','direction_reset','full_objective_available'}
    strings={'status':'incomplete','phase':'serialized_selected','exception_record_file':'evaluation-200002-failure.json'}
    def value(k):
        if k=='metrics':return metrics
        if k.endswith('sha256'):return 'f'*64
        if k in strings:return strings[k]
        if k in bools:return False
        return big
    rows=[]
    for node in ast.walk(tree):
        if not (isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='append'):continue
        name=node.args[0].value;record=node.args[1]
        if isinstance(record,ast.Dict):keys=[k.value for k in record.keys]
        else:
            assert isinstance(record,ast.Call) and record.func.id=='dict'
            keys=identity_keys+[x.arg for x in record.keywords]
        row={k:value(k) for k in keys};size=len((json.dumps(row)+'\n').encode());bound=1024 if name=='EVALUATIONS.jsonl' else 6144
        assert size<=bound,(name,size,bound)
        rows.append({'stream':name,'serialized_bytes':size,'allowance_bytes':bound,'keys':keys})
    assert len(rows)==5 and len(metric_keys)>=24
    return {'passed':True,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'helper_sha256':hashlib.sha256(helper.read_bytes()).hexdigest(),'rows':rows,'metric_keys':metric_keys,'exception_details':'Full exception text retained separately in a single failure JSON; its bytes are covered by the 64 MiB miscellaneous allowance, not the fixed event row.','no_compression_credit':True}
if __name__=='__main__':print(json.dumps(check(),indent=2))
