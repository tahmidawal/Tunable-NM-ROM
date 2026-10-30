"""Exercise actual nested evaluation event retention with a tiny NumPy stand-in."""
import ast,json,time
from pathlib import Path
import numpy as np
from accuracy_v2.train import digest

def check():
    tree=ast.parse(Path(__file__).with_name('trace_refine.py').read_text())
    node=next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name=='full')
    for i,x in enumerate(node.body):
        if isinstance(x,ast.Nonlocal):node.body[i]=ast.Global(names=x.names)
    module=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
    records=[];calls=[]
    def guard(final):
        calls.append(final)
        if len(calls)==2:raise TimeoutError('injected mid-rule deadline')
    def calc(t,x,w,y,g,nv,ng,importance):
        return (float(len(x)),{'base_objective':float(len(x)),'combined_objective':float(len(x))}),np.ones_like(t)*len(x)
    env=dict(np=np,jnp=np,time=time,digest=digest,evaluations=0,initiated_evaluations=0,accepted=0,started=time.monotonic(),n=3,config={'block_points':2},x=np.zeros((3,2)),w=np.ones(3),target=np.zeros((3,2,4)),gradient=np.zeros((3,2,2,4)),nv=np.ones(4),ng=np.ones(4),importance=np.ones(4)/4,guard=guard,calc=calc,append=lambda name,row:records.append((name,row)))
    exec(compile(module,'actual_nested_full','exec'),env)
    try:env['full'](np.zeros(5),phase='line_search')
    except TimeoutError:pass
    else:raise AssertionError('expected injected interruption')
    assert [r[1]['status'] for r in records]==['pending','incomplete']
    assert records[-1][1]['completed_blocks']==1 and records[-1][1]['full_objective_available'] is False
    assert env['evaluations']==0 and env['initiated_evaluations']==1
    env['guard']=lambda final:None
    f,g,metrics=env['full'](np.zeros(5),final=True,phase='serialized_selected')
    assert f==3 and np.array_equal(g,np.full(5,3)) and env['evaluations']==1 and env['initiated_evaluations']==2
    assert records[-1][1]['status']=='completed' and records[-2][1]['status']=='pending'
    return {'passed':True,'source_function':'actual trace_refine.full AST','mid_rule_interruption_retained':True,'partial_sum_not_full_measurement':True,'completed_evaluation_retained':True,'actual_training_or_native_work':False}
if __name__=='__main__':print(json.dumps(check(),indent=2))
