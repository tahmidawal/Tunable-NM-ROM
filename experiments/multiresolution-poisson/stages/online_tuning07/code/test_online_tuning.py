"""Bounded controls for newly active stationarity stop and source-only multistart."""
from tuning_core import *
from tuning_summary import summarize,select
import json

p,z,_=sc.load_pkl(Path(__file__).resolve().parent/'runs/pilot04/checkpoints/original_relative.pkl' if not Path('in/model.pkl').exists() else Path('in/model.pkl'))
ops=assemble(p,z,32,64,150);cache=weak_code_cache(ops,z);source=full_source(32,source_params(7090703,6)[0])
limits=dict(stationarity_tolerance=1e-6,linear_backward_error_limit=1e-12)
def run(budget=150,tau=.01,stop=None,starts=1):
    preset=dict(budget=budget,tau=tau,stationarity_stop=stop,starts=starts)
    return tuning_query(source,ops,cache,preset,make_tuning_kernel(ops,preset,1e-12),limits)
old=speed_query(source,ops,cache,.01,make_speed_kernel(ops,150,'skinny_sine_products','nearest_cached_scaled_weak_prediction'),'skinny_sine_products','nearest_cached_scaled_weak_prediction')
new=run();assert relative(old[0],new[0])<1e-9 and old[1]['reason']==new[1]['reason']
initial=run(stop=1.);assert initial[1]['reason']==6 and initial[1]['attempts']==0
budget=run(budget=1,tau=0.);assert budget[1]['reason']==0 and budget[1]['attempts']==1
multi=run(budget=3,tau=0.,starts=4);rr=multi[1]
assert rr['selected_start']==int(np.argmin([s['residual'] for s in rr['starts']]))
fm=np.asarray(ops['project'](jnp.asarray(source),ops['S'],ops['I'],ops['J'],ops['W']))
nearest=np.argsort(np.sum((np.asarray(cache['predictions'])-fm)**2,axis=1))[:4]
assert [s['selected_training_code_index'] for s in rr['starts']]==nearest.tolist()
for _,row in (new,initial,budget,multi):assert abs(row['total_seconds']-row['input_seconds']-row['fused_device_seconds']-row['output_seconds'])<1e-10
print(json.dumps(dict(passed=True,baseline_field_relative=relative(old[0],new[0]),initial_stationarity_reason=initial[1]['reason'],budget_reason=budget[1]['reason'],multistart_chosen=rr['selected_start']),indent=2))
