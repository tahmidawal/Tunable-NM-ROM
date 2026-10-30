"""Same-process CPU paired timing. Callables return the very outputs scored later.

The launcher must pin the CPU core/thread policy for NGSolve and NumPy. This utility does
not turn heterogeneous device timings or different output contracts into a fair comparison.
"""
from __future__ import annotations
import copy
import hashlib
import os
import time
import numpy as np


def outputs_sha256(outputs):
    digest=hashlib.sha256()
    for name,value in sorted(outputs.items()):
        a=np.ascontiguousarray(value)
        if a.dtype.kind not in "fiu" or not np.isfinite(a).all():
            raise ValueError("finite numeric invocation outputs required")
        digest.update(name.encode());digest.update(str((a.shape,str(a.dtype))).encode());digest.update(a.tobytes())
    return digest.hexdigest()


def paired_cpu(calls,contract,cores,repetitions=7,burn_seconds=.25):
    return _timed_cpu(calls,contract,cores,repetitions,burn_seconds,2)


def calibration_cpu(call,contract,cores,repetitions=7,burn_seconds=.25):
    record=_timed_cpu({'candidate':call},contract,cores,repetitions,burn_seconds,1)
    record['comparison_kind']='one candidate in the same calibration allocation; no cross-job comparison'
    return record


def _timed_cpu(calls,contract,cores,repetitions,burn_seconds,minimum_methods):
    if repetitions!=7 or not 1<=cores<=16 or contract not in ("full_velocity_pressure_fields","fixed_probes","integrated_forces"):
        raise ValueError("unregistered timing contract")
    if any(os.environ.get(key)!=str(cores) for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS")):
        raise RuntimeError("launcher must freeze matching CPU thread limits before imports")
    names=list(calls)
    if len(names)<minimum_methods:
        raise ValueError("paired comparison needs multiple methods")
    warmup=[]
    for name in names:
        try:
            record=calls[name]()
            warmup.append({'method':name,'query_failed':False,'work':copy.deepcopy(record.get('work',{}))})
        except Exception as exc:
            warmup.append({'method':name,'query_failed':True,'exception':repr(exc)})
    rows=[];retained_outputs=[]
    for rep in range(repetitions):
        # Explicit CPU clock/cache burn immediately before the timed block.
        a=np.full((128,128),.001,dtype=np.float64);b=np.eye(128,dtype=np.float64)
        burn_start=time.perf_counter()
        while time.perf_counter()-burn_start<burn_seconds:
            b=a@b+np.eye(128)
        order=names if rep%2==0 else names[::-1]
        for name in order:
            start=time.perf_counter();cpu_start=time.process_time()
            try:
                record=calls[name]()
            except Exception as exc:
                partial=getattr(exc,'invocation',None)
                if isinstance(partial,dict):
                    record=dict(partial)
                    record.update(query_failed=True,exception=repr(exc))
                    if record.get('outputs') is None and record.get('outputs_available') is False:
                        record['outputs']={}  # Explicitly unavailable, never fabricated values.
                else:
                    record={'output_contract':contract,'outputs':{},'work':{'status':'query_failed'},
                            'query_failed':True,'exception':repr(exc),
                            'failure_evidence':'call raised; no returned state or output was available'}
            cpu=time.process_time()-cpu_start;elapsed=time.perf_counter()-start
            if record.get("output_contract")!=contract or "outputs" not in record or "work" not in record:
                raise ValueError("query did not return its output contract, actual outputs and work")
            # Callers may reuse FE/NumPy output buffers and mutable accounting dictionaries.
            # Snapshot EVERYTHING before another invocation can overwrite this evidence.
            evidence_start=time.perf_counter()
            snapshot=copy.deepcopy(record)
            for key,value in snapshot["outputs"].items():
                array=np.asarray(value)
                array.setflags(write=False)
                snapshot["outputs"][key]=array
            checksum=outputs_sha256(snapshot["outputs"])
            evidence_seconds=time.perf_counter()-evidence_start
            rows.append({"repetition":rep,"method":name,"elapsed_seconds":elapsed,"cpu_seconds":cpu,
                         "output_sha256":checksum,"work":snapshot["work"],
                         "query_failed":bool(snapshot.get("query_failed",False)),
                         "component_seconds":snapshot.get("component_seconds",{}),"order":list(order),
                         "evidence_snapshot_seconds":evidence_seconds})
            retained_outputs.append({"repetition":rep,"method":name,"outputs":snapshot["outputs"],
                                     "invocation":snapshot})
    return {"contract":contract,"cores":cores,"device":"CPU","warmups_discarded":len(warmup),"warmup_attempts":warmup,
            "repetitions":rows,"actual_invocation_outputs":retained_outputs,"burn_seconds":burn_seconds,
            "evidence_snapshot_timing":"reported separately after query; caller-required output copies belong inside query"}
