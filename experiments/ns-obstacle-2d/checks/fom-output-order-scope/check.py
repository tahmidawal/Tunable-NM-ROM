"""Execute actual cache closure/order assignments with a boundary-rule sentinel."""
from pathlib import Path
import ast,copy,hashlib,json,subprocess,types
W=Path(__file__).resolve().parents[4];relative='experiments/ns-obstacle-2d/fom_outputs.py';before_commit='e887cfab0';before=subprocess.check_output(['git','-C',str(W),'show',before_commit+':'+relative],text=True);after=(W/relative).read_text()
expected=before.replace('batches=[];order=[];elements=', 'batches=[];native_order=[];elements=').replace('batches.append(batch);order.extend(indices)', 'batches.append(batch);native_order.extend(indices)').replace('native_to_requested=np.asarray(order,dtype=np.int64)', 'native_to_requested=np.asarray(native_order,dtype=np.int64)')
assert after==expected
class Observed(Exception):pass
class NativeSentinel:
 BND='boundary';VOL='volume'
 def IntegrationRule(self,element_type,order):raise Observed(type(order).__name__,order if isinstance(order,int) else len(order))
class Mesh:
 def Elements(self,region):return [types.SimpleNamespace(type='SEGMENT')]
 def GetTrafo(self,element):return None

def probe(source,alias):
 tree=ast.parse(source);factory=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='factory');cache=next(n for n in factory.body if isinstance(n,ast.FunctionDef) and n.name=='cache')
 assignments=[n for n in ast.walk(factory) if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)]
 order=next(n for n in assignments if n.targets[0].id=='order' and isinstance(n.value,ast.Subscript));mapping=next(n for n in assignments if n.targets[0].id==alias and isinstance(n.value,ast.List))
 fn=ast.parse('def probe(cfg,ng,mesh):\n    pass').body[0];fn.body=[copy.deepcopy(order),copy.deepcopy(cache),copy.deepcopy(mapping),ast.parse('cache(True)').body[0]]
 ns={};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'<actual-cache-scope>','exec'),ns)
 try:ns['probe']({'diagnostic_order':12},NativeSentinel(),Mesh())
 except Observed as exc:return {'boundary_rule_order_type':exc.args[0],'value_or_list_length':exc.args[1]}
 raise AssertionError('boundary IntegrationRule sentinel not reached')
a=probe(before,'order');b=probe(after,'native_order');assert a['boundary_rule_order_type']=='list' and b=={'boundary_rule_order_type':'int','value_or_list_length':12}
out={'schema':'ns-obstacle-output-quadrature-order-scope-regression-v1','passed':True,'before_commit':before_commit,'before_sha256':hashlib.sha256(before.encode()).hexdigest(),'after_sha256':hashlib.sha256(after.encode()).hexdigest(),'before':a,'after':b,'only_changes':'three mapping-local identifier replacements; physics/coordinates/rules/thresholds unchanged','scope':'actual Python closure and assignments; fake boundary sentinel, no native evaluation or submission'}
Path(__file__).with_name('RESULT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
