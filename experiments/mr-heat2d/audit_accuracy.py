"""Independent NumPy/SciPy audit of saved heat comparison fields and operators."""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess
import pickle
from scipy.special import expit

import numpy as np
import scipy.linalg
import scipy.fft


def audit(record):
    audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    archive = record/"archive"; out = archive/"outputs"
    result = json.loads((out/"results.json").read_text())
    assert result["complete"] and result["metadata"]["backend"] == "gpu"
    assert result["metadata"]["x64"] and result["metadata"]["precision"] == "highest"
    log = (archive/f'job-{result["metadata"]["job_id"]}.log').read_text()
    assert "jax_backend=gpu" in log and "ACCURACY HEAT COMPLETE" in log
    for forbidden in ("Captured constant", "captured constant", "large constant", "RESOURCE_EXHAUSTED", "Out of memory", "No space left", "Traceback"):
        assert forbidden not in log, forbidden
    source = result["source_manifest"]
    for path, expected in source["sha256"].items():
        blob = (archive/path).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == expected
        if path in source["provenance"]:
            origin = source["provenance"][path]
            saved = subprocess.check_output(["git", "show", origin["commit"]+":"+origin["git_path"]])
            assert saved == blob
    checkpoint = pickle.loads((archive/(result["settings"]["checkpoint"]+".pkl")).read_bytes())
    params = checkpoint["params"]
    models = {}
    for model in result["models"]:
        blob = (out/model["path"]).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == model["sha256"]
        item = pickle.loads(blob)
        for key in ("B","g","out_scale"):
            def leaves(value):
                if isinstance(value,(list,tuple)):
                    for x in value: yield from leaves(x)
                else: yield value
            for a,b in zip(leaves(params[key]),leaves(item["params"][key])): np.testing.assert_array_equal(a,b)
        assert item["config"] == result["config"]
        models[model["name"]] = item
        if model['name']=='frozen':
            for key in params:
                for a,b in zip(leaves(params[key]),leaves(item['params'][key])): np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(checkpoint['codes'],item['codes'])
        else:
            assert model['details']['updates']==result['settings']['updates']
            assert model['details']['sampled_snapshots']==result['settings']['updates']*result['settings']['batch_size']
    def draws(seed,count):
        rng=np.random.default_rng(seed); cfg=result['config']
        return np.column_stack((rng.uniform(*cfg['center_range'],(count,2)),rng.uniform(*cfg['width_range'],count),rng.uniform(*cfg['amplitude_range'],count)))
    expected_cases=[]
    for cohort in result['settings']['cohorts']:
        expected_cases.extend((cohort['name'],draw) for draw in draws(cohort['seed'],cohort['count']))
    assert len(expected_cases)==len(result['cases'])
    for (name,draw),case in zip(expected_cases,result['cases']):
        assert name==case['cohort']; np.testing.assert_array_equal(draw,case['draw'])
    training=dict(np.load(out/'training_compression.npz'))
    expected_training=np.concatenate((draws(result['config']['train_seed'],result['config']['n_train']),draws(790713,128)))
    np.testing.assert_array_equal(expected_training,training['training_draws'])
    assert not any(np.array_equal(a,b['draw']) for a in expected_training for b in result['cases'])
    def mlp(layers, x):
        for w, b in layers[:-1]:
            x = x@w+b; x = x*expit(x)
        w, b = layers[-1]
        return x@w+b
    def head(z): return mlp(params["h"], z)+z@params["h_lin"]
    def head_jacobian(z):
        x = z; jac = np.eye(len(z))
        for w, b in params["h"][:-1]:
            x = x@w+b; sig = expit(x)
            jac = (sig*(1+x*(1-sig)))[:, None]*(w.T@jac)
            x = x*sig
        w, b = params["h"][-1]
        return w.T@jac+params["h_lin"].T
    axis = np.arange(1, 64)/64
    xy = np.stack(np.meshgrid(axis, axis, indexing="ij"), -1).reshape(-1, 2)
    angle = 2*np.pi*(xy@params["B"])
    features = np.concatenate((np.sin(angle), np.cos(angle)), -1)
    mask = 16*xy[:,0]*(1-xy[:,0])*xy[:,1]*(1-xy[:,1])
    sampled_bank = (params["out_scale"]*mask)[:, None]*mlp(params["g"], features)
    manifold_checks = []; weak_checks = []; initial_checks = []
    initial_targets = {}; operators = {}
    checked = set(); largest_delta = 0.; metric_count = 0
    @lru_cache(maxsize=8)
    def load(path):
        a = np.load(out/path)["field"]
        assert a.dtype == np.float64 and np.isfinite(a).all()
        return a
    def field(spec):
        a = load(spec["path"])
        if spec["path"] not in checked:
            digest = hashlib.sha256(str((a.shape, a.dtype.str)).encode()+np.ascontiguousarray(a).tobytes()).hexdigest()
            assert digest == spec["sha256_array"]
            assert list(a.shape) == spec["shape"] and a.dtype.str == spec["dtype"]
            checked.add(spec["path"])
        return a
    def check_metrics(predicted, truth, n, saved):
        nonlocal largest_delta, metric_count
        a = predicted.reshape(len(predicted), -1); b = truth.reshape(len(truth), -1)
        diff = np.linalg.norm(a-b, axis=1); norm = np.linalg.norm(b, axis=1)
        actual = dict(relative_current=diff/np.maximum(norm, 1e-300), relative_initial=diff/max(norm[0], 1e-300),
                      absolute_l2=diff/n, truth_norm_over_initial=norm/max(norm[0], 1e-300),
                      energy=np.sum(a*a, axis=1)/(2*n*n),
                      state_change_from_initial=np.linalg.norm(a-a[0], axis=1)/max(np.linalg.norm(a[0]), 1e-300))
        for key, value in actual.items():
            delta = float(np.max(np.abs(value-np.asarray(saved[key]))))
            largest_delta = max(largest_delta, delta); metric_count += len(value)
            np.testing.assert_allclose(value, saved[key], atol=2e-13, rtol=2e-13)
        assert np.array_equal(norm/max(norm[0], 1e-300)<1e-3, saved["vanished_below_1e-3_initial"])
    max_reference_delta = 0.
    for case in result["cases"]:
        coarse, fine = field(case["reference_coarse"]), field(case["reference_fine"])
        check_metrics(coarse, fine, max(result["settings"]["requested_intervals"]), case["reference_refinement"])
        max_reference_delta = max(max_reference_delta, max(case["reference_refinement"]["relative_current"]))
    refs = {(r["intervals"], r["case"]): r for r in result["case_fields"]}
    def generated_initial(n,draw):
        axis=np.arange(1,n,dtype=np.float64)/n
        x,y=axis[:,None],axis[None,:]; cx,cy,width,amplitude=draw
        return amplitude*16*x*(1-x)*y*(1-y)*np.exp(-((x-cx)**2+(y-cy)**2)/(2*width**2))
    source_checks=[]; reference_checks=[]
    largest=max(result['settings']['requested_intervals'])
    for (n,cid),spec in refs.items():
        saved=field(spec['initial']); expected=generated_initial(n,result['cases'][cid]['draw'])
        relative=float(np.linalg.norm(saved-expected)/np.linalg.norm(expected))
        assert relative<1e-13
        source_checks.append(dict(intervals=n,case=cid,relative_difference=relative))
    # Independent NumPy/SciPy reconstruction of both continuum-spectral references
    # from the recorded seed/draw, at their original meshes, before restriction.
    for case in result['cases']:
        for nf,key in zip(result['settings']['reference_pair'],['reference_coarse','reference_fine']):
            initial=generated_initial(nf,case['draw']); spectrum=scipy.fft.dstn(initial,type=1,norm='ortho')
            eig=(np.pi*np.arange(1,nf,dtype=np.float64))**2
            lam=eig[:,None]+eig[None,:]; saved=field(case[key]); stride=nf//largest
            errors=[]
            for index,time_value in enumerate(result['config']['times']):
                predicted=initial if index==0 else scipy.fft.idstn(spectrum*np.exp(-result['config']['diffusivity']*time_value*lam),type=1,norm='ortho')
                predicted=predicted[stride-1::stride,stride-1::stride]
                relative=float(np.linalg.norm(predicted-saved[index])/np.linalg.norm(saved[index]))
                assert relative<1e-11
                errors.append(relative)
            reference_checks.append(dict(case=case['case'],reference_intervals=nf,relative_differences=errors))
        print('independent_reference',case['case'],flush=True)
    linear_rows = {(r["intervals"], r["case"]): r for r in result["rows"] if r["method"] == "linear_weak_exact"}
    for n in result["settings"]["requested_intervals"]:
        op = dict(np.load(out/"assembly"/f"n{n}.npz")); operators[n] = op
        # Recover the independent orthogonal projection target from the linear
        # control's initial field on common physical nodes. Full-field metric
        # checks below cover that control. No JAX or production solver import.
        sampled_q = scipy.linalg.solve_triangular(op["triangular"].T, sampled_bank.T, lower=True).T
        q, upper = np.linalg.qr(sampled_q, mode="reduced")
        for cid in range(len(result["cases"])):
            linear = field(linear_rows[n,cid]["repetitions"][0]["field"])[0]
            stride = n//64
            samples = linear[stride-1::stride, stride-1::stride].reshape(-1)
            target = scipy.linalg.solve_triangular(upper,q.T@samples)
            assert np.linalg.norm(sampled_q@target-samples)/np.linalg.norm(samples)<1e-10
            initial_targets[n,cid] = target
    cg_checks = []
    @lru_cache(maxsize=4)
    def exact_discrete(n,cid,theta,dt):
        initial = field(refs[n,cid]['initial'])
        k = np.arange(1,n,dtype=np.float64)
        eig1 = 4*n*n*np.sin(np.pi*k/(2*n))**2
        eigenvalues = eig1[:,None]+eig1[None,:]
        nu = result['config']['diffusivity']
        factor = (1-(1-theta)*dt*nu*eigenvalues)/(1+theta*dt*nu*eigenvalues)
        spectrum = scipy.fft.dstn(initial,type=1,norm='ortho')
        values = [initial]
        for time in result['config']['times'][1:]:
            exponent = round(time/dt)
            assert abs(exponent*dt-time)<1e-12
            values.append(scipy.fft.idstn(spectrum*factor**exponent,type=1,norm='ortho'))
        return np.asarray(values)
    counts = {}; summaries = []
    for row in result["rows"]:
        key = row["intervals"], row["case"], row["method"]
        assert key not in counts; counts[key] = len(row["repetitions"])
        assert counts[key] == result["settings"]["timing_repetitions"]
        refs_row = refs[key[:2]]; initial = field(refs_row["initial"])
        truth, discrete = field(refs_row["physical"]), field(refs_row["discrete"])
        for rep in row["repetitions"]:
            pred = field(rep["field"])
            if rep["solver"]:
                params = models[row["method"].removeprefix("nmrom_")]["params"]
                zs = np.asarray(rep["solver"]["latents"])
                assert zs.shape == (len(result["config"]["times"]), result["config"]["k"])
                predicted_samples = head(zs)@sampled_bank.T
                stride = row["intervals"]//64
                actual_samples = pred[:, stride-1::stride, stride-1::stride].reshape(pred.shape[0], -1)
                err = float(np.linalg.norm(predicted_samples-actual_samples)/np.linalg.norm(actual_samples))
                assert err < 1e-10
                manifold_checks.append(err)
                op = operators[row['intervals']]
                matrix = op['matrix']; initial_matrix = op['triangular']
                internal = np.asarray(rep['solver']['internal_latents'])
                initial_zs = np.asarray(rep['solver']['initial_latents'])
                dt = result['settings']['dt']; nu = result['config']['diffusivity']
                nsteps = round(result['config']['times'][-1]/dt)
                stride_t = round((result['config']['times'][1]-result['config']['times'][0])/dt)
                assert internal.shape == (nsteps+1,result['config']['k'])
                assert initial_zs.shape == (2,result['config']['k'])
                np.testing.assert_array_equal(zs,internal[::stride_t])
                infos = np.asarray(rep['solver']['initial_fits'])
                np.testing.assert_array_equal(internal[0],initial_zs[np.argmin(infos[:,3])])
                def diagnostics(matrix,target,z,saved):
                    scale = max(np.linalg.norm(target),1e-14)
                    residual = (matrix@head(z)-target)/scale
                    jac = matrix@head_jacobian(z)/scale
                    pair = np.array([np.linalg.norm(residual),np.linalg.norm(jac.T@residual)/max(np.linalg.norm(jac),1e-30)])
                    np.testing.assert_allclose(pair,np.asarray(saved)[[3,4]],rtol=1e-8,atol=1e-11)
                    if saved[2] == 1: assert pair[1] <= result['settings']['gradient_tolerance']+1e-11
                    return float(np.max(np.abs(pair-np.asarray(saved)[[3,4]])))
                for z,info in zip(initial_zs,infos):
                    initial_checks.append(diagnostics(initial_matrix,initial_targets[key[:2]],z,info))
                factor = (1-dt*nu*op['mode_lam']/2)/(1+dt*nu*op['mode_lam']/2)
                assert len(rep['solver']['steps']) == nsteps
                for i,info in enumerate(rep['solver']['steps']):
                    weak_checks.append(diagnostics(matrix,factor*(matrix@head(internal[i])),internal[i+1],info))
            if row['method'] in result['settings']['cg_methods']:
                spec = result['settings']['cg_methods'][row['method']]
                infos = np.asarray(rep['cg_steps'])
                nsteps = round(result['config']['times'][-1]/spec['dt'])
                assert infos.shape == (nsteps,4) and np.isfinite(infos).all()
                assert np.all((infos[:,0]>=0)&(infos[:,0]<=spec['max_iterations']))
                np.testing.assert_array_equal(infos[:,0],infos[:,0].astype(int))
                expected = infos[:,2]<=spec['tolerance']*(1+1e-6)+1e-12
                np.testing.assert_array_equal(expected,infos[:,3].astype(bool))
                discrepancy = np.linalg.norm((pred-exact_discrete(row['intervals'],row['case'],spec['theta'],spec['dt'])).reshape(len(pred),-1),axis=1)
                scale = np.linalg.norm(exact_discrete(row['intervals'],row['case'],spec['theta'],spec['dt']).reshape(len(pred),-1),axis=1)
                delta = discrepancy/np.maximum(scale,1e-300)
                cg_checks.append(dict(intervals=row['intervals'],case=row['case'],method=row['method'],repetition=rep['repetition'],
                    relative_current_vs_exact_time_discrete=delta.tolist(),
                    max_recurrent_true_difference=float(np.max(np.abs(infos[:,1]-infos[:,2]))),
                    max_true_relative_residual=float(np.max(infos[:,2])),
                    nonconverged_steps=int(np.sum(infos[:,3]!=1)),total_iterations=int(np.sum(infos[:,0])),
                    zero_iteration_steps=int(np.sum(infos[:,0]==0))))
            else: assert 'cg_steps' not in rep
            check_metrics(pred, truth, row["intervals"], rep["vs_physical"])
            check_metrics(pred, discrete, row["intervals"], rep["vs_same_grid"])
            phases = rep["phases"]
            assert min(phases.values()) > 0
            assert abs(phases["host_seconds"]-sum(phases[k] for k in ("input_seconds", "device_seconds", "output_seconds"))) < 1e-12
            if row["method"].startswith("fom_"): np.testing.assert_array_equal(pred[0], initial)
            if "refined_field" in rep:
                check_metrics(pred, field(rep["refined_field"]), row["intervals"], rep["vs_half_step"])
        if row['method'] == result['settings']['methods'][-1]:
            print('audited_case', row['intervals'], row['case'], flush=True)
    ns = result["settings"]["requested_intervals"]
    assert len(counts) == len(ns)*len(result["cases"])*len(result["settings"].get("methods",list({r["method"] for r in result["rows"]})))
    assert sum(counts.values()) == result["timed_invocations"]
    operator_errors = []
    for n in ns:
        op = np.load(out/"assembly"/f"n{n}.npz")
        # QR least squares is independent of the production SVD-based pinverse.
        r = op["triangular"]; c = scipy.linalg.solve_triangular(r.T, op["matrix"].T, lower=True).T
        q, upper = np.linalg.qr(c, mode="reduced")
        nu = result["config"]["diffusivity"]; lam = op["mode_lam"]; dt = result["settings"]["dt"]
        generator = -nu*scipy.linalg.solve_triangular(upper, q.T@(lam[:, None]*c))
        cn = scipy.linalg.solve_triangular(upper, q.T@(((1-dt*nu*lam/2)/(1+dt*nu*lam/2))[:, None]*c))
        for name, a, b in (("generator", generator, op["generator"]), ("cn_step", cn, op["cn_step"])):
            relative = float(np.linalg.norm(a-b)/np.linalg.norm(b))
            assert relative < 1e-10
            operator_errors.append(dict(intervals=n, operator=name, relative_error=relative))
        assert np.max(scipy.linalg.eigvals(generator).real) < 0
        grams = np.load(out/'assembly'/f'grams_n{n}.npz')
        for name,matrix in [('initial_gram',r),('weak_gram',op['matrix'])]:
            relative = float(np.linalg.norm(matrix.T@matrix-grams[name])/np.linalg.norm(grams[name]))
            assert relative < 1e-12
            operator_errors.append(dict(intervals=n,operator=name,relative_error=relative))
        for name in sorted({r["method"] for r in result["rows"]}):
            rows = [r for r in result["rows"] if r["intervals"] == n and r["method"] == name]
            reps = [rep for row in rows for rep in row["repetitions"]]
            summary = dict(intervals=n, method=name, cases=len(rows), invocations=len(reps),
                worst_physical_error=max(max(r["vs_physical"]["relative_current"]) for r in reps),
                worst_initial_error=max(r["vs_physical"]["relative_current"][0] for r in reps),
                worst_same_grid_error=max(max(r["vs_same_grid"]["relative_current"]) for r in reps),
                largest_energy_increase=max(float(np.max(np.diff(r["vs_physical"]["energy"]))) for r in reps))
            for contract in ("device", "host"):
                key = contract+"_seconds"
                values = [rep["phases"][key] for rep in reps]
                summary[contract+"_median_ms"] = float(np.median(values)*1000)
                summary[contract+"_outliers"] = int(sum(sum(rep["phases"][key]>1.5*np.median([r["phases"][key] for r in row["repetitions"]]) for rep in row["repetitions"]) for row in rows))
            summary["nonstationary_fit_count"] = sum(int(np.any((np.asarray(rep["solver"]["initial_fits"])[:, 2] != 1) & (np.asarray(rep["solver"]["initial_fits"])[:, 2] != -1))) for rep in reps if rep["solver"])
            summary["nonstationary_step_count"] = sum(int(np.sum(np.asarray(rep["solver"]["steps"])[:, 2] != 1)) for rep in reps if rep["solver"])
            summary["total_initial_attempts"] = sum(sum(int(x[0]) for x in rep["solver"]["initial_fits"]) for rep in reps if rep["solver"])
            summary["total_step_attempts"] = sum(sum(int(x[0]) for x in rep["solver"]["steps"]) for rep in reps if rep["solver"])
            summary["skipped_second_initializations"] = sum(int(rep["solver"]["initial_fits"][1][2] == -1) for rep in reps if rep["solver"])
            if name == "linear_weak_cn":
                summary["worst_half_step_current_delta"] = max(max(r["vs_half_step"]["relative_current"]) for r in reps)
            if name in result['settings']['cg_methods']:
                checks = [c for c in cg_checks if c['intervals']==n and c['method']==name]
                summary.update(cg_nonconverged_steps=sum(c['nonconverged_steps'] for c in checks),
                    cg_total_iterations=sum(c['total_iterations'] for c in checks),
                    cg_zero_iteration_steps=sum(c['zero_iteration_steps'] for c in checks),
                    cg_max_true_relative_residual=max(c['max_true_relative_residual'] for c in checks),
                    cg_max_recurrent_true_difference=max(c['max_recurrent_true_difference'] for c in checks),
                    worst_exact_time_discrete_error=max(max(c['relative_current_vs_exact_time_discrete']) for c in checks))
            summaries.append(summary)
    nmrom_parity = []
    prior_out = record.parent/"iterative_cg09/archive/outputs"
    if prior_out.exists() and len(result["cases"]) >= 12:
        prior = json.loads((prior_out/"results.json").read_text())
        old = {(r["intervals"],r["case"]):r for r in prior["rows"] if r["method"] == "nmrom_cholesky"}
        for row in result["rows"]:
            if row["method"] != "nmrom_frozen" or row["case"] >= 12: continue
            a=field(row["repetitions"][0]["field"])
            b=np.load(prior_out/old[row["intervals"],row["case"]]["repetitions"][0]["field"]["path"])["field"]
            relative=float(np.linalg.norm(a-b)/np.linalg.norm(b)); assert relative < 1e-9
            nmrom_parity.append(dict(intervals=row["intervals"],case=row["case"],relative_difference=relative))
    reconstruction = result["reconstruction"]
    truth=field(reconstruction["truth"])
    projected=field(reconstruction["bank_projection"])
    for i,metrics in enumerate(reconstruction["bank_errors"]): check_metrics(projected[i],truth[i],reconstruction["intervals"],metrics)
    representation_summaries=[]
    # Reconstruct the training-mesh bank independently and audit all diagnostic
    # fields, every selected fit's objective/gradient, and orthogonal projection.
    triangular=operators[64]["triangular"]
    qbank=scipy.linalg.solve_triangular(triangular.T,sampled_bank.T,lower=True).T
    np.testing.assert_allclose(qbank.T@qbank,np.eye(result['config']['r']),atol=1e-10,rtol=1e-10)
    training_metrics=[]
    for model in result['models']:
        params=models[model['name']]['params']; codes=models[model['name']]['codes']
        residual=head(codes)@training['triangular'].T-training['target']
        errors2=(np.sum(residual**2,axis=1)+training['perpendicular2'])/training['norm2']
        actual=dict(full_relative_mse=float(np.mean(errors2)),initial_relative_mse=float(np.mean(errors2[::6])),
                    later_relative_mse=float(np.mean(errors2[np.arange(len(errors2))%6!=0])))
        expected=result['training_baseline'] if model['name']=='frozen' else model['details']
        for key,value in actual.items(): np.testing.assert_allclose(value,expected[key],rtol=1e-10,atol=1e-13)
        training_metrics.append(dict(model=model['name'],**actual))
    flat=truth.reshape(-1,63*63); targets=flat@qbank
    np.testing.assert_allclose((targets@qbank.T).reshape(truth.shape),projected,rtol=1e-10,atol=1e-12)
    for record_fit in reconstruction["models"]:
        params=models[record_fit["name"]]["params"]
        predicted=field(record_fit["reconstructed"])
        zs=np.asarray(record_fit["latents"]); infos=np.asarray(record_fit["fits"]); best=np.asarray(record_fit["best"])
        np.testing.assert_allclose((head(zs)@sampled_bank.T).reshape(predicted.shape),predicted,rtol=1e-10,atol=1e-12)
        selected=infos[np.arange(len(infos)),best]
        fit_gradient_deltas=[]
        for target,z,saved in zip(targets,zs,selected):
            scale=max(np.linalg.norm(target),1e-14)
            residual=(triangular@head(z)-target)/scale; jac=triangular@head_jacobian(z)/scale
            actual=[np.linalg.norm(residual),np.linalg.norm(jac.T@residual)/max(np.linalg.norm(jac),1e-30)]
            np.testing.assert_allclose(actual,saved[[3,4]],rtol=1e-7,atol=1e-10)
            fit_gradient_deltas.append(float(np.max(np.abs(actual-saved[[3,4]]))))
        for i,metrics in enumerate(record_fit["metrics"]): check_metrics(predicted[i],truth[i],reconstruction["intervals"],metrics)
        representation_summaries.append(dict(model=record_fit["name"],
            worst_initial_error=max(m["relative_current"][0] for m in record_fit["metrics"]),
            worst_later_error=max(max(m["relative_current"][1:]) for m in record_fit["metrics"]),
            nonstationary_selected=int(np.sum(selected[:,2] != 1)),
            maximum_gradient_difference=max(fit_gradient_deltas)))
    fine_summaries=[]
    assert result['bank_gate']['passed']
    fine_records=result['fine_reconstruction']
    assert len(fine_records)==len(result['cases'])
    for diagnostic in fine_records:
        n,cid=diagnostic['intervals'],diagnostic['case']; ref=refs[n,cid]
        exact=field(ref['discrete']); physical=field(ref['physical'])
        projected=field(diagnostic['bank_projection']); targets=field(diagnostic['target'])
        check_metrics(projected,exact,n,diagnostic['bank_same_grid'])
        check_metrics(projected,physical,n,diagnostic['bank_physical'])
        op=operators[n]; sampled_q=scipy.linalg.solve_triangular(op['triangular'].T,sampled_bank.T,lower=True).T
        stride=n//64
        np.testing.assert_allclose(targets@sampled_q.T,projected[:,stride-1::stride,stride-1::stride].reshape(len(exact),-1),rtol=1e-9,atol=1e-11)
        for fit in diagnostic['models']:
            params=models[fit['name']]['params']; predicted=field(fit['reconstructed'])
            all_zs=np.asarray(fit['all_latents']); infos=np.asarray(fit['fits']); best=np.asarray(fit['best'])
            zs=np.asarray(fit['latents'])
            np.testing.assert_array_equal(best,np.argmin(infos[:,:,3],axis=1))
            np.testing.assert_array_equal(zs,all_zs[np.arange(len(zs)),best])
            np.testing.assert_allclose(head(zs)@sampled_bank.T,predicted[:,stride-1::stride,stride-1::stride].reshape(len(exact),-1),rtol=1e-9,atol=1e-11)
            max_delta=0.
            for target,zstarts,infostarts in zip(targets,all_zs,infos):
                for z,saved in zip(zstarts,infostarts):
                    scale=max(np.linalg.norm(target),1e-14)
                    residual=(op['triangular']@head(z)-target)/scale; jac=op['triangular']@head_jacobian(z)/scale
                    actual=np.array([np.linalg.norm(residual),np.linalg.norm(jac.T@residual)/max(np.linalg.norm(jac),1e-30)])
                    np.testing.assert_allclose(actual,saved[[3,4]],rtol=1e-7,atol=1e-10)
                    max_delta=max(max_delta,float(np.max(np.abs(actual-saved[[3,4]]))))
            check_metrics(predicted,exact,n,fit['vs_same_grid']); check_metrics(predicted,physical,n,fit['vs_physical'])
            fine_summaries.append(dict(intervals=n,case=cid,model=fit['name'],
                initial_error=fit['vs_same_grid']['relative_current'][0],later_worst=max(fit['vs_same_grid']['relative_current'][1:]),
                nonstationary_selected=int(np.sum(infos[np.arange(len(zs)),best,2]!=1)),maximum_diagnostic_difference=max_delta))
    output = dict(passed=True, audit_source_sha256=audit_source_sha256, result_sha256=hashlib.sha256((out/"results.json").read_bytes()).hexdigest(),
        metadata=result["metadata"], source_commit=source["source_commit"], settings=result["settings"],
        unique_fields_checked=len(checked), timed_invocations_checked=sum(counts.values()),
        metric_entries_checked=metric_count, maximum_metric_difference=largest_delta,
        maximum_reference_refinement=max_reference_delta, operator_errors=operator_errors,
        independent_source_checks=source_checks,independent_reference_checks=reference_checks,training_metric_checks=training_metrics,
        outlier_rule="Above 1.5 times the median of the same case/method/mesh repetition group; none excluded.",
        summaries=summaries, representation_summaries=representation_summaries, fine_representation_summaries=fine_summaries, prior_nmrom_field_parity=nmrom_parity,
        nonlinear_decoder_checks=len(manifold_checks), maximum_sampled_decoder_error=max(manifold_checks),
        time_step_weak_checks=len(weak_checks), maximum_time_step_weak_difference=max(weak_checks),
        initial_fit_weak_checks=len(initial_checks), maximum_initial_fit_weak_difference=max(initial_checks),
        initial_target_audit="Recovered by independent QR from sampled linear initial projection; not a full-grid independent projection reconstruction.",
        cg_checks=cg_checks,
        cg_reference="Independent NumPy/SciPy sine transform of the exact matching finite-difference theta-method; every CG output checked. Full internal CG fields are not stored for independent per-step true-residual reconstruction.")
    analysis = record/"analysis"; analysis.mkdir(exist_ok=True)
    (analysis/"audit.json").write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps({k:v for k,v in output.items() if k not in ("algebra_parity","prior_nmrom_field_parity","cg_checks")}, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("record", type=Path)
    audit(parser.parse_args().record)
