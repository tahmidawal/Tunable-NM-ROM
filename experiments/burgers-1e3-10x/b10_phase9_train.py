#!/usr/bin/env python
"""Phase-9 arm-generic exact-full-grid Burgers nonlinear-ROM trainer.

Scientific execution is cluster-only.  ``--smoke`` is deliberately synthetic,
small, and writes no locked experiment artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import time
import warnings

import jax
jax.config.update("jax_enable_x64", True)
from jax import lax
import jax.numpy as jnp
import numpy as np
import optax

import b10_common as c
import b10_phase3 as p3
import b10_phase4 as p4
import b10_phase7 as p7
import b10_phase7_train as p7train
import b10_phase8_d as p8
import b10_s0_spline as base
import b10_spline as spline
import b10_spline_train as legacy


F64 = jnp.float64
TRAIN_MIX = ((64, 0, 512), (128, 0, 128), (256, 0, 64))
SELECTION_MIX = ((64, 512, 64), (128, 512, 32), (256, 512, 16))
N_ORDER = (64, 128, 256)
BATCH_BY_N = {64: 8, 128: 2, 256: 1}
BATCHES_PER_N = 3264
PHASES = {
    "encoder": {"epochs": 9, "seed": 311011, "lr": (1e-3, 1e-4)},
    "joint": {"epochs": 27, "seed": 11, "lr": (1e-3, 1e-5)},
    "predictor": {"epochs": 18, "seed": 100011, "lr": (5e-4, 5e-6)},
}
IDENTITY_TOL = 2e-14
AFFINE_ATOL = 2e-15
TRUST_ATTEMPTS = 40
DELTA0, DELTA_MIN, DELTA_MAX = .25, 2.0 ** -20, 1.0
LAMBDA0, LAMBDA_MIN, LAMBDA_MAX = 1e-6, 1e-12, 1e12
ACCEPT_RHO = 1e-4

ARMS = {
    "T1": {"label": "G1/q19", "channels": 16, "residual": False, "q": 19,
           "state": 24, "generator_count": 30594, "encoder_count": 29811,
           "predictor_count": 2104},
    "T2": {"label": "G2/q32", "channels": 32, "residual": True, "q": 32,
           "state": 37, "generator_count": 165954, "encoder_count": 164384,
           "predictor_count": 2533},
}

EXPECTED = dict(p8.EXPECTED)
EXPECTED.update({
    "p8_json": "be0ca15e5c36f45a1f9a8fdfe86b79a592b530c069ae6d25c1f573c59c974f4f",
    "p8_npz": "93f8eabcaeaafd419e868a3a7f74ddbd84429c23345570a5f46d5fa38af2471f",
    "p8_audit": "d1e9e7bd238ac9e9320afc018fa5ff85afcd51d6ff714a61eb47c3b32900de17",
    "p8_manifest": "1af89411e6a37c1106fa9285e50468bd936e34a7ab15b070b06d1a82846416ad",
})

warnings.filterwarnings("error", message=r"(?i).*captured.*large.*constant.*", category=Warning)


def tree_count(tree):
    return int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(tree)))


def atomic_json(path, value):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".partial"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=1, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_pickle(path, value):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".partial"
    with open(temporary, "wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def array_sha(value):
    value = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(np.asarray(value.shape, np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _draw_factory(seed, arm_index):
    root = jax.random.fold_in(jax.random.PRNGKey(seed), arm_index)
    index = 0
    def draw(shape, fan_in, fan_out, bias=False):
        nonlocal index
        key = jax.random.fold_in(root, index); index += 1
        if bias:
            return jnp.zeros(shape, F64)
        return math.sqrt(2.0 / (fan_in + fan_out)) * jax.random.normal(key, shape, F64)
    return draw


def init_generator(config, seed=11):
    channels, qdim = config["channels"], config["q"]
    draw = _draw_factory(seed, 1 if qdim == 19 else 2)
    def dense(height, width):
        outputs = height * width * channels
        return {"W": draw((qdim, outputs), qdim, outputs),
                "b": draw((outputs,), qdim, outputs, True)}
    def conv():
        fan = 9 * channels
        return {"W": draw((3, 3, channels, channels), fan, fan),
                "b": draw((channels,), fan, fan, True)}
    depth = 2 if config["residual"] else 1
    tree = {
        "coarse_seed": dense(6, 6), "fine_seed": dense(4, 4),
        "coarse_blocks": tuple(tuple(conv() for _ in range(depth)) for _ in range(3)),
        "fine_blocks": tuple(tuple(conv() for _ in range(depth)) for _ in range(3)),
        "coarse_out": {"W": draw((1, 1, channels, 1), channels, 1),
                       "b": draw((1,), channels, 1, True)},
        "fine_out": {"W": draw((1, 1, channels, 1), channels, 1),
                     "b": draw((1,), channels, 1, True)},
    }
    if tree_count(tree) != config["generator_count"]:
        raise AssertionError("generator parameter count")
    return tree


def _conv(value, layer, reflect=False):
    if reflect:
        value = jnp.pad(value, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="reflect")
    return lax.conv_general_dilated(value, layer["W"], (1, 1), "VALID",
                                    dimension_numbers=("NHWC", "HWIO", "NHWC")) + layer["b"]


def _generator_head(q, seed, blocks, output, height, width, residual):
    channels = blocks[0][0]["b"].size
    value = (q @ seed["W"] + seed["b"]).reshape((-1, height, width, channels))
    for block in blocks:
        value = jnp.repeat(jnp.repeat(value, 2, axis=1), 2, axis=2)
        if residual:
            hidden = jax.nn.swish(_conv(value, block[0], True))
            value = jax.nn.swish(value + _conv(hidden, block[1], True))
        else:
            value = jax.nn.swish(_conv(value, block[0], True))
    return _conv(value, output)[..., 0]


def apply_generator(parameters, q, mean, scales, config):
    q = jnp.asarray(q, F64); prefix = q.shape[:-1]; flat = q.reshape((-1, config["q"]))
    coarse = _generator_head(flat, parameters["coarse_seed"], parameters["coarse_blocks"],
                             parameters["coarse_out"], 6, 6, config["residual"]).reshape((-1, 2304))
    fine = _generator_head(flat, parameters["fine_seed"], parameters["fine_blocks"],
                           parameters["fine_out"], 4, 4, config["residual"]).reshape((-1, 1024))
    out = jnp.concatenate((coarse * scales[0], fine * scales[1]), axis=1) + mean
    return out.reshape(prefix + (3328,))


def init_encoder(config, seed=11):
    channels, qdim = config["channels"], config["q"]
    # Disjoint from generator and invariant under host process ordering.
    draw = _draw_factory(seed, 101 if qdim == 19 else 102)
    def conv(cin, cout, kernel):
        return {"W": draw((kernel, kernel, cin, cout), kernel*kernel*cin, kernel*kernel*cout),
                "b": draw((cout,), kernel*kernel*cin, kernel*kernel*cout, True)}
    depth = 2 if config["residual"] else 1
    def head():
        return {"input": conv(1, channels, 1),
                "blocks": tuple(tuple(conv(channels, channels, 3) for _ in range(depth))
                                for _ in range(3))}
    hidden = 52 * channels
    tree = {"coarse": head(), "fine": head(),
            "output": {"W": draw((hidden, qdim), hidden, qdim),
                       "b": draw((qdim,), hidden, qdim, True)}}
    if tree_count(tree) != config["encoder_count"]:
        raise AssertionError((tree_count(tree), config["encoder_count"]))
    return tree


def _encoder_head(value, parameters, residual):
    value = _conv(value, parameters["input"])
    for block in parameters["blocks"]:
        if residual:
            hidden = jax.nn.swish(_conv(value, block[0], True))
            value = jax.nn.swish(value + _conv(hidden, block[1], True))
        else:
            value = jax.nn.swish(_conv(value, block[0], True))
        value = lax.reduce_window(value, 0.0, lax.add, (1,2,2,1), (1,2,2,1), "VALID") / 4.0
    return value.reshape((value.shape[0], -1))


def apply_encoder(parameters, normalized, config):
    normalized = jnp.asarray(normalized, F64)
    coarse = normalized[:, :2304].reshape((-1,48,48,1))
    fine = normalized[:, 2304:].reshape((-1,32,32,1))
    hidden = jnp.concatenate((_encoder_head(coarse, parameters["coarse"], config["residual"]),
                              _encoder_head(fine, parameters["fine"], config["residual"])), axis=1)
    raw = 3.0 * jnp.tanh((hidden @ parameters["output"]["W"] + parameters["output"]["b"]) / 3.0)
    return raw, jnp.tanh(raw)


def init_predictor(config, seed=12):
    draw = _draw_factory(seed, 200 + config["q"])
    dims = (7, 32, 32, config["state"])
    layers = tuple({"W": draw((a,b), a,b), "b": draw((b,),a,b,True)} for a,b in zip(dims[:-1],dims[1:]))
    if tree_count(layers) != config["predictor_count"]:
        raise AssertionError("predictor parameter count")
    return layers


def apply_predictor(parameters, features):
    x = features
    for layer in parameters[:-1]:
        x = jax.nn.swish(x @ layer["W"] + layer["b"])
    return jnp.tanh(x @ parameters[-1]["W"] + parameters[-1]["b"])


def decode_cox(generator, states, coords, mask, mean, scales, config):
    coefficients = apply_generator(generator, states[:, 5:], mean, scales, config)
    return jax.vmap(lambda s, a: p4.decode_one_cox_jax(s, a, coords, mask, p7.H1))(states, coefficients)


def normalized_generated(generator, q, mean, scales, config):
    values = apply_generator(generator, q, mean, scales, config)
    return jnp.concatenate(((values[:, :2304]-mean[:2304])/scales[0],
                            (values[:, 2304:]-mean[2304:])/scales[1]), axis=1)


def exact_loss(generator, states, truth, coords, mask, mean, scales, config):
    prediction = decode_cox(generator, states, coords, mask, mean, scales, config)
    numer = jnp.sum(jnp.square(prediction-truth), axis=1)
    denom = jnp.maximum(jnp.sum(jnp.square(truth), axis=1), 1e-300)
    return jnp.mean(numer/denom), prediction


def schedule_permutations(phase, epochs, sizes):
    out = {}
    for epoch in range(1, epochs+1):
        rng = np.random.Generator(np.random.PCG64(PHASES[phase]["seed"] + epoch))
        for n in N_ORDER:
            out[f"{phase}_e{epoch}_N{n}"] = rng.permutation(sizes[n]).astype(np.int32)
    return out


def schedule_lr(start, stop, steps):
    return optax.cosine_decay_schedule(start-stop, steps, alpha=0.0) if stop == 0 else (
        lambda step: stop + (start-stop) * .5 * (1+jnp.cos(jnp.pi*jnp.minimum(step,steps)/steps)))


def optimizer(lr):
    return optax.chain(optax.clip_by_global_norm(1.0),
                       optax.adamw(lr, b1=.9, b2=.999, eps=1e-8, weight_decay=1e-6))


def validate_chains(args, arm):
    bindings = {}
    for phase in ("p4", "p5", "p6"):
        for kind in ("json", "npz", "audit", "manifest"):
            key = f"{phase}_{kind}"; path = getattr(args, key)
            if c.sha256(path) != EXPECTED[key]:
                raise SystemExit(f"immutable {key} mismatch")
            bindings[key] = {"basename": os.path.basename(path), "sha256": EXPECTED[key]}
    for kind in ("json", "npz", "checkpoint", "audit", "manifest"):
        key = f"p7_{kind}"; path = getattr(args, key)
        if c.sha256(path) != EXPECTED[key]:
            raise SystemExit(f"immutable {key} mismatch")
        bindings[key] = {"basename": os.path.basename(path), "sha256": EXPECTED[key]}
    for kind in ("json", "npz", "audit", "manifest"):
        key = f"p8_{kind}"; path = getattr(args, key)
        if c.sha256(path) != EXPECTED[key]:
            raise SystemExit(f"immutable {key} mismatch")
        bindings[key] = {"basename": os.path.basename(path), "sha256": EXPECTED[key]}
    for phase in ("p4", "p5", "p6", "p7", "p8"):
        audit = p8.load_json(getattr(args, f"{phase}_audit"))
        if audit.get("status") != "pass" or audit.get("negative_aware") is not True:
            raise SystemExit(f"{phase} audit not passing")
    reports = {phase: p8.load_json(getattr(args, f"{phase}_json"))
               for phase in ("p4", "p5", "p6", "p7", "p8")}
    if not (reports["p4"]["decision"]["selected_spatial_arm"] == "H1"
            and reports["p5"]["train_targets"]["integrity_pass"] is True
            and reports["p5"]["train_targets"]["snapshot_count"] == 35904
            and reports["p6"]["decision"]["repair_licensed"] is True
            and reports["p7"]["decision"]["g1_seed11_pass"] is False
            and reports["p8"]["decision"]["p8_d_valid"] is False
            and reports["p8"]["gates"]["health"] is False):
        raise SystemExit("P4-P8 scientific decision chain mismatch")
    if arm == "T2":
        if any(x is None for x in (args.t1_json,args.t1_npz,args.t1_checkpoint,args.t1_audit,args.t1_manifest)):
            raise SystemExit("T2 requires immutable audited T1 capacity license")
        t1 = p8.load_json(args.t1_json); audit = p8.load_json(args.t1_audit)
        if not (audit.get("status") == "pass" and audit.get("negative_aware") is True
                and audit.get("decision", {}).get("g2_licensed") is True
                and t1.get("decision", {}).get("g2_licensed") is True
                and audit.get("source_json_sha256")==c.sha256(args.t1_json)
                and audit.get("source_npz_sha256")==c.sha256(args.t1_npz)
                and audit.get("checkpoint_sha256")==c.sha256(args.t1_checkpoint)
                and audit.get("manifest_sha256")==c.sha256(args.t1_manifest)
                and t1.get("npz",{}).get("sha256")==c.sha256(args.t1_npz)
                and t1.get("checkpoint",{}).get("sha256")==c.sha256(args.t1_checkpoint)
                and audit.get("decision")==t1.get("decision")):
            raise SystemExit("T2 capacity license absent")
        for name, path in (("t1_json", args.t1_json), ("t1_npz", args.t1_npz),
                           ("t1_checkpoint",args.t1_checkpoint),("t1_audit", args.t1_audit),("t1_manifest",args.t1_manifest)):
            bindings[name] = {"basename": os.path.basename(path), "sha256": c.sha256(path)}
    bindings["prereg"] = {"basename": os.path.basename(args.prereg), "sha256": c.sha256(args.prereg)}
    return reports, bindings


def load_populations(args, smoke):
    legacy.SMOKE = smoke
    if smoke:
        train = p7train.smoke_datasets("train")
        selection = p7train.smoke_datasets("selection")
        rng = np.random.default_rng(92011)
        coefficients = rng.normal(0, .05, (4,3328))
        records = [{"basename":"synthetic-only", "sha256":None, "snapshot_count":4, "N":16}]
        return train, selection, coefficients, records
    train, selection = legacy.load_mix(TRAIN_MIX, "train"), legacy.load_mix(SELECTION_MIX, "selection")
    p5r = p8.load_json(args.p5_json)
    coefficients, physical, features, records = p8.load_train_targets(args, p5r)
    p8.bind_regeneration(train, selection, physical, features,
                         {name: value for name, value in np.load(args.p7_npz, allow_pickle=False).items()})
    return train, selection, coefficients, records


def train_normalization(coefficients, features):
    started = time.perf_counter()
    mean = np.mean(coefficients, axis=0, dtype=np.float64)
    centered = coefficients - mean
    scales = np.asarray((np.sqrt(np.mean(centered[:,:2304]**2, dtype=np.float64)),
                         np.sqrt(np.mean(centered[:,2304:]**2, dtype=np.float64))), np.float64)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise SystemExit("invalid train-only normalization")
    normalized = np.concatenate((centered[:,:2304]/scales[0], centered[:,2304:]/scales[1]), axis=1)
    feature_mean, feature_scale, empirical = legacy.predictor_feature_statistics(features)
    return {"mean":mean, "scales":scales, "normalized":normalized,
            "feature_mean":feature_mean, "feature_scale":feature_scale,
            "feature_empirical":empirical, "elapsed_s":time.perf_counter()-started,
            "host_bytes":int(sum(x.nbytes for x in (mean,scales,normalized,feature_mean,feature_scale,empirical))),
            "source_indices":np.arange(len(coefficients),dtype=np.int32)}


def per_n_views(datasets, normalized):
    views = {}; offset = 0
    for item in datasets:
        count = item["flat"].shape[0]
        views[item["N"]] = {"truth":np.asarray(item["flat"],np.float64),
                            "affine":np.asarray(item["affine"],np.float64),
                            "features":np.asarray(item["features"],np.float64),
                            "normalized":np.asarray(normalized[offset:offset+count],np.float64),
                            "coords":np.asarray(item["coords"],np.float64),
                            "mask":np.asarray(item["mask"],np.float64),
                            "global":np.arange(offset,offset+count,dtype=np.int32), "item":item}
        offset += count
    return views


def make_update_kernels(config, total_steps):
    lr_encoder = schedule_lr(*PHASES["encoder"]["lr"], total_steps["encoder"])
    lr_joint = schedule_lr(*PHASES["joint"]["lr"], total_steps["joint"])
    lr_predictor = schedule_lr(*PHASES["predictor"]["lr"], total_steps["predictor"])
    opt_encoder, opt_joint, opt_predictor = optimizer(lr_encoder), optimizer(lr_joint), optimizer(lr_predictor)

    def encoder_loss(params, normalized, truth, affine, coords, mask, mean, scales):
        qraw,q = apply_encoder(params["encoder"], normalized, config)
        states = jnp.concatenate((affine,q),axis=1)
        field,_ = exact_loss(params["generator"],states,truth,coords,mask,mean,scales,config)
        generated = normalized_generated(params["generator"],q,mean,scales,config)
        coeff = .5*jnp.mean((generated[:,:2304]-normalized[:,:2304])**2) + .5*jnp.mean((generated[:,2304:]-normalized[:,2304:])**2)
        reg = 1e-6*jnp.mean(qraw**2)
        return field+.01*coeff+reg, jnp.asarray((field,coeff,reg))
    def joint_loss(params, take, normalized, truth, affine, coords, mask, mean, scales):
        raw = params["q_raw"][take]; q=jnp.tanh(raw); states=jnp.concatenate((affine,q),axis=1)
        field,_=exact_loss(params["generator"],states,truth,coords,mask,mean,scales,config)
        generated=normalized_generated(params["generator"],q,mean,scales,config)
        coeff=.5*jnp.mean((generated[:,:2304]-normalized[:,:2304])**2)+.5*jnp.mean((generated[:,2304:]-normalized[:,2304:])**2)
        reg=1e-6*jnp.mean(raw**2)
        return field+.01*coeff+reg,jnp.asarray((field,coeff,reg))
    def predictor_loss(pred, generator, features, target_states, truth, coords, mask, mean, scales):
        states=apply_predictor(pred,features)
        field,_=exact_loss(generator,states,truth,coords,mask,mean,scales,config)
        state=jnp.mean((states-target_states)**2)
        return field+.1*state,jnp.asarray((field,state))

    @jax.jit
    def encoder_update(params,state,normalized,truth,affine,coords,mask,mean,scales):
        (loss,parts),grad=jax.value_and_grad(encoder_loss,has_aux=True)(params,normalized,truth,affine,coords,mask,mean,scales)
        updates,state=opt_encoder.update(grad,state,params)
        return optax.apply_updates(params,updates),state,loss,parts
    @jax.jit
    def joint_update(params,state,take,normalized,truth,affine,coords,mask,mean,scales):
        (loss,parts),grad=jax.value_and_grad(joint_loss,has_aux=True)(params,take,normalized,truth,affine,coords,mask,mean,scales)
        updates,state=opt_joint.update(grad,state,params)
        return optax.apply_updates(params,updates),state,loss,parts
    @jax.jit
    def predictor_update(pred,state,generator,features,target_states,truth,coords,mask,mean,scales):
        (loss,parts),grad=jax.value_and_grad(predictor_loss,has_aux=True)(pred,generator,features,target_states,truth,coords,mask,mean,scales)
        updates,state=opt_predictor.update(grad,state,pred)
        return optax.apply_updates(pred,updates),state,loss,parts
    return (opt_encoder,opt_joint,opt_predictor),(encoder_update,joint_update,predictor_update)


def evaluate_full(datasets, generator, states, mean, scales, config, arrays, prefix, k3=False):
    report={"meshes":{}}; pooled=[]; all_num=[]; all_den=[]; identities=[]; boundaries=[]
    offset=0
    for item in datasets:
        count=len(item["flat"]); one=np.asarray(states[offset:offset+count]); offset += count
        evaluate=jax.jit(lambda gen,st,xy,ma,me,sc: decode_cox(gen,st,xy,ma,me,sc,config))
        prediction=[]
        batch=max(1,min(8,count))
        for start in range(0,count,batch):
            prediction.append(np.asarray(evaluate(generator,jnp.asarray(one[start:start+batch]),
                                                   jnp.asarray(item["coords"]),jnp.asarray(item["mask"]),
                                                   jnp.asarray(mean),jnp.asarray(scales))))
        prediction=np.concatenate(prediction)
        truth=item["flat"]; diff=prediction-truth
        num=np.sum(diff**2,axis=1); den=np.maximum(np.sum(truth**2,axis=1),1e-300)
        case_num=num.reshape(item["case_count"],item["num_times"]).sum(axis=1)
        case_den=den.reshape(item["case_count"],item["num_times"]).sum(axis=1)
        trajectory=np.sqrt(case_num/np.maximum(case_den,1e-300))
        boundary_count=np.sum(prediction[:,item["mask"]==0] != 0.0,axis=1).astype(np.int32)
        identity=np.zeros(count,np.float64)
        # K3 is charged and checked only at terminal; Cox remains the scientific field.
        if k3:
            padded=((len(item["coords"])+127)//128)*128
            coords=np.pad(item["coords"],((0,padded-len(item["coords"])),(0,0)))
            mask=np.pad(item["mask"],(0,padded-len(item["mask"])))
            decoder=p4.make_pallas_hierarchical_decoder(batch,padded,p7.H1)
            coarse=jnp.asarray(p3.span_polynomial_table_np(48),F64)
            fine=jnp.asarray(p3.span_polynomial_table_np(32),F64)
            @jax.jit
            def evaluator(gen,st,xy,ma,me,sc,coarse_arg,fine_arg):
                coeff=apply_generator(gen,st[:,5:],me,sc,config)
                return decoder(st,coeff,xy,ma,coarse_arg,fine_arg)
            for start in range(0,count,batch):
                take=min(batch,count-start); st=one[start:start+take]
                if take<batch: st=np.concatenate((st,np.repeat(st[-1:],batch-take,axis=0)))
                actual=np.asarray(evaluator(generator,jnp.asarray(st),jnp.asarray(coords),jnp.asarray(mask),jnp.asarray(mean),jnp.asarray(scales),coarse,fine))[:take,:len(item["coords"])]
                identity[start:start+take]=np.linalg.norm(actual-prediction[start:start+take],axis=1)/np.maximum(np.linalg.norm(prediction[start:start+take],axis=1),1e-300)
        arrays[f"{prefix}_N{item['N']}_numerator"] = num
        arrays[f"{prefix}_N{item['N']}_denominator"] = den
        arrays[f"{prefix}_N{item['N']}_trajectory"] = trajectory
        arrays[f"{prefix}_N{item['N']}_boundary_count"] = boundary_count
        arrays[f"{prefix}_N{item['N']}_identity"] = identity
        report["meshes"][str(item["N"])]= {"trajectory_error_mean":float(np.mean(trajectory)),
            "trajectory_error_worst":float(np.max(trajectory)), "all_finite":bool(np.all(np.isfinite(prediction))),
            "boundary_violation_count":int(np.sum(boundary_count)),
            "k3_cox_identity_worst":float(np.nanmax(identity)) if k3 else None}
        pooled.extend(trajectory); all_num.extend(num); all_den.extend(den); identities.extend(identity); boundaries.extend(boundary_count)
    pooled=np.asarray(pooled)
    report["pooled"]={"trajectory_error_mean":float(np.mean(pooled)),"trajectory_error_worst":float(np.max(pooled)),
        "mean_snapshot_relative_l2_squared":float(np.mean(np.asarray(all_num)/np.asarray(all_den))),
        "all_finite":bool(all(row["all_finite"] for row in report["meshes"].values())),
        "boundary_violation_count":int(np.sum(boundaries)),
        "k3_cox_identity_worst":float(np.nanmax(identities)) if k3 else None}
    return report


def encode_batches(encoder, normalized, config, batch=128):
    fn=jax.jit(lambda enc,x: apply_encoder(enc,x,config))
    raw=[]
    for start in range(0,len(normalized),batch):
        raw.append(np.asarray(fn(encoder,jnp.asarray(normalized[start:start+batch]))[0]))
    return np.concatenate(raw)


def predictor_batches(predictor, features, batch=256):
    fn=jax.jit(apply_predictor); out=[]
    for start in range(0,len(features),batch):
        out.append(np.asarray(fn(predictor,jnp.asarray(features[start:start+batch]))))
    return np.concatenate(out)


def gate(metrics, mean_limit, worst_limit, terminal=False):
    rows=list(metrics["meshes"].values())+[metrics["pooled"]]
    return bool(all(row["trajectory_error_mean"] <= mean_limit
                    and row["trajectory_error_worst"] <= worst_limit
                    and row["all_finite"] and row["boundary_violation_count"] == 0
                    and (not terminal or row["k3_cox_identity_worst"] <= IDENTITY_TOL)
                    for row in rows))


def trust_health_gate(health):
    return bool(all(bool(value) for key,value in health.items()
                    if key not in ("breakdown_count","unhealthy_exhaustion_count")))


def preflight(config, views, variables, optimizers, kernels, norm, smoke, job_started):
    """Compile and time no-update work, always discarding returned parameters."""
    elapsed_before_preflight=float(time.perf_counter()-job_started)
    repeats = 1 if smoke else 13
    warmups = 0 if smoke else 3
    medians={}; raw={}; before={name:array_sha(np.concatenate([np.ravel(np.asarray(x)) for x in jax.tree_util.tree_leaves(tree)]))
                            for name,tree in variables.items()}
    params={"generator":variables["generator"],"encoder":variables["encoder"]}
    states=(optimizers[0].init(params), optimizers[1].init({"generator":variables["generator"],"q_raw":variables["q_raw"]}),
            optimizers[2].init(variables["predictor"]))
    for n,view in views.items():
        take=np.arange(min(BATCH_BY_N.get(n,2),len(view["truth"])),dtype=np.int32)
        common=(jnp.asarray(view["truth"][take]),jnp.asarray(view["affine"][take]),
                jnp.asarray(view["coords"]),jnp.asarray(view["mask"]),jnp.asarray(norm["mean"]),jnp.asarray(norm["scales"]))
        args=(
            (params,states[0],jnp.asarray(view["normalized"][take]),*common),
            ({"generator":variables["generator"],"q_raw":variables["q_raw"]},states[1],
             jnp.asarray(view["global"][take]),jnp.asarray(view["normalized"][take]),*common),
            (variables["predictor"],states[2],variables["generator"],
             jnp.asarray((view["features"][take]-norm["feature_mean"])/norm["feature_scale"]),
             jnp.asarray(variables["target_states"][view["global"][take]]),common[0],common[2],common[3],common[4],common[5]),
        )
        for name,kernel,one_args in zip(("encoder","joint","predictor"),kernels,args):
            times=[]
            for rep in range(repeats):
                started=time.perf_counter(); result=kernel(*one_args)
                jax.block_until_ready(result); elapsed=time.perf_counter()-started
                if rep>=warmups: times.append(elapsed)
            raw[f"{name}_N{n}"]=times; medians[f"{name}_N{n}"]=float(np.median(times))
        if not smoke:
            eval_fn=jax.jit(lambda gen,st,tr,xy,ma,me,sc: exact_loss(gen,st,tr,xy,ma,me,sc,config))
            eval_states=jnp.asarray(variables["target_states"][view["global"][take]])
            trust_fn=make_trust_attempt(config); q0=eval_states[:1,5:]; aff0=jnp.asarray(view["affine"][take[:1]])
            truth0=jnp.asarray(view["truth"][take[:1]]); objective0=jnp.asarray((1.,),F64)
            capacity_fn=make_q_diagnostic(config)
            extra=(
                ("evaluation",lambda:eval_fn(variables["generator"],eval_states,common[0],common[2],common[3],common[4],common[5])),
                ("trust",lambda:trust_fn(variables["generator"],q0,aff0,truth0,common[2],common[3],common[4],common[5],
                                          jnp.asarray((DELTA0,)),jnp.asarray((LAMBDA0,)),jnp.asarray((True,)),objective0)),
                ("capacity",lambda:capacity_fn(variables["generator"],q0,aff0,truth0,common[2],common[3],common[4],common[5])),
            )
            for name,invoke in extra:
                times=[]
                for rep in range(repeats):
                    start=time.perf_counter(); result=invoke(); materialized=jax.tree_util.tree_map(lambda x:np.asarray(jax.device_get(x)),result); elapsed=time.perf_counter()-start
                    if rep>=warmups: times.append(elapsed)
                raw[f"{name}_N{n}"]=times; medians[f"{name}_N{n}"]=float(np.median(times))
    # Exact deterministic count; evaluation/trust are bounded by conservative
    # multiples of the slowest full-grid gradient unit.
    update_seconds=sum(medians[f"{phase}_N{n}"]*BATCHES_PER_N*PHASES[phase]["epochs"]
                       for phase in PHASES for n in views if n in N_ORDER)
    if smoke:
        eval_seconds=trust_seconds=capacity_seconds=0.0
    else:
        eval_seconds=56*sum(math.ceil(len(v["truth"])/BATCH_BY_N[n])*medians[f"evaluation_N{n}"] for n,v in views.items())
        selection_counts={64:3264,128:1632,256:816}
        trust_seconds=2*TRUST_ATTEMPTS*sum(selection_counts[n]*medians[f"trust_N{n}"] for n in N_ORDER)
        capacity_seconds=sum(len(v["truth"])*medians[f"capacity_N{n}"] for n,v in views.items())
    audit_regeneration_reserve=0.0 if smoke else elapsed_before_preflight
    projected=update_seconds+eval_seconds+max(trust_seconds,2*capacity_seconds)+audit_regeneration_reserve
    after={name:array_sha(np.concatenate([np.ravel(np.asarray(x)) for x in jax.tree_util.tree_leaves(tree)]))
           for name,tree in variables.items()}
    actual_elapsed=float(time.perf_counter()-job_started); allocation=float(os.environ.get("SLURM_TIMELIMIT_SECONDS",57600)); remaining=max(0.,allocation-actual_elapsed)
    return {"raw_seconds":raw,"median_seconds":medians,"projected_update_seconds":float(update_seconds),
            "projected_evaluation_seconds":float(eval_seconds),"projected_trust_seconds":float(trust_seconds),"projected_capacity_seconds":float(capacity_seconds),
            "elapsed_before_preflight_seconds":elapsed_before_preflight,"audit_regeneration_reserve_seconds":audit_regeneration_reserve,
            "projected_terminal_seconds":float(projected),"safety_factor":1.15,
            "allocation_seconds":allocation,"actual_elapsed_at_decision_seconds":actual_elapsed,
            "actual_remaining_at_decision_seconds":remaining,"required_with_safety_seconds":float(1.15*projected),
            "proceed_before_update1":bool(1.15*projected<=remaining),
            "weights_bitwise_unchanged":before==after,"before_hashes":before,"after_hashes":after}


def training_batches(view, permutation):
    batch=BATCH_BY_N[view["item"]["N"]]
    return permutation.reshape((-1,batch))


def train_all(config, datasets, views, norm, args, smoke, job_started):
    sizes={n:len(v["truth"]) for n,v in views.items()}
    epochs={name:(1 if smoke else row["epochs"]) for name,row in PHASES.items()}
    permutations = ({name:np.arange(next(iter(sizes.values())),dtype=np.int32)
                     for name in ("encoder_e1_N16","joint_e1_N16","predictor_e1_N16")}
                    if smoke else schedule_permutations("encoder",epochs["encoder"],sizes)
                    | schedule_permutations("joint",epochs["joint"],sizes)
                    | schedule_permutations("predictor",epochs["predictor"],sizes))
    total_steps={phase:epochs[phase]*(1 if smoke else BATCHES_PER_N*3) for phase in PHASES}
    optimizers,kernels=make_update_kernels(config,total_steps)
    generator=init_generator(config); encoder=init_encoder(config); predictor=init_predictor(config)
    q_raw=encode_batches(encoder,norm["normalized"],config)
    target_states=np.concatenate((legacy.concatenate(datasets,"affine"),np.tanh(q_raw)),axis=1)
    variables={"generator":generator,"encoder":encoder,"predictor":predictor,"q_raw":jnp.asarray(q_raw),
               "target_states":target_states}
    pre=preflight(config,views,variables,optimizers,kernels,norm,smoke,job_started)
    permitted=bool(smoke or (pre["weights_bitwise_unchanged"] and pre["proceed_before_update1"]))
    if not permitted:
        return variables,permutations,{},pre,False
    params={"generator":generator,"encoder":encoder}; state=optimizers[0].init(params)
    history={}; arrays={}; global_step=0; final_optimizer_states={}
    for phase in ("encoder","joint","predictor"):
        if phase=="joint":
            q_raw=encode_batches(params["encoder"],norm["normalized"],config)
            arrays["encoder_handoff_q_raw"]=q_raw.copy()
            variables["q_raw"]=jnp.asarray(q_raw)
            joint={"generator":params["generator"],"q_raw":variables["q_raw"]}; state=optimizers[1].init(joint)
        if phase=="predictor":
            generator=joint["generator"]; q_raw=np.asarray(joint["q_raw"])
            target_states=np.concatenate((legacy.concatenate(datasets,"affine"),np.tanh(q_raw)),axis=1)
            predictor=variables["predictor"]; state=optimizers[2].init(predictor)
        for epoch in range(1,epochs[phase]+1):
            last=None
            ncycle=(tuple(views) if smoke else N_ORDER)
            per_n={n:training_batches(v,permutations[f"{phase}_e{epoch}_N{n}"]) if not smoke
                   else [np.arange(min(2,len(v["truth"])),dtype=np.int32)] for n,v in views.items()}
            batches=1 if smoke else BATCHES_PER_N
            for b in range(batches):
                for n in ncycle:
                    v=views[n]; local=np.asarray(per_n[n][b]); take=v["global"][local]
                    truth=jnp.asarray(v["truth"][local]); affine=jnp.asarray(v["affine"][local]);
                    coords=jnp.asarray(v["coords"]); mask=jnp.asarray(v["mask"])
                    if phase=="encoder":
                        params,state,loss,parts=kernels[0](params,state,jnp.asarray(v["normalized"][local]),truth,affine,coords,mask,jnp.asarray(norm["mean"]),jnp.asarray(norm["scales"]))
                    elif phase=="joint":
                        joint,state,loss,parts=kernels[1](joint,state,jnp.asarray(take),jnp.asarray(v["normalized"][local]),truth,affine,coords,mask,jnp.asarray(norm["mean"]),jnp.asarray(norm["scales"]))
                    else:
                        standardized=(v["features"][local]-norm["feature_mean"])/norm["feature_scale"]
                        predictor,state,loss,parts=kernels[2](predictor,state,generator,jnp.asarray(standardized),jnp.asarray(target_states[take]),truth,coords,mask,jnp.asarray(norm["mean"]),jnp.asarray(norm["scales"]))
                    global_step += 1; last=(float(loss),np.asarray(parts).tolist())
            if phase=="encoder":
                q=encode_batches(params["encoder"],norm["normalized"],config); generator_eval=params["generator"]
                states_eval=np.concatenate((legacy.concatenate(datasets,"affine"),np.tanh(q)),axis=1)
            elif phase=="joint":
                generator_eval=joint["generator"]; q=np.asarray(joint["q_raw"])
                states_eval=np.concatenate((legacy.concatenate(datasets,"affine"),np.tanh(q)),axis=1)
            else:
                generator_eval=generator
                folded=legacy.fold_predictor_standardization(predictor,norm["feature_mean"],norm["feature_scale"])
                states_eval=predictor_batches(folded,legacy.concatenate(datasets,"features"))
            metrics=evaluate_full(datasets,generator_eval,states_eval,norm["mean"],norm["scales"],config,arrays,f"{phase}_epoch{epoch}",False)
            history[f"{phase}_epoch{epoch}"]={"terminal_update":global_step,"last_batch":last,"metrics":metrics}
            atomic_json(args.progress_json,{"status":"in_progress","phase":phase,"epoch":epoch,
                "global_update":global_step,"scientific_metrics_exposed":False})
            atomic_pickle(args.work_checkpoint,{"phase":phase,"epoch":epoch,"global_update":global_step,
                "generator":jax.tree_util.tree_map(np.asarray,generator_eval),"encoder":jax.tree_util.tree_map(np.asarray,params["encoder"]),
                "predictor":jax.tree_util.tree_map(np.asarray,predictor if phase=="predictor" else variables["predictor"]),
                "q_raw":np.asarray(joint["q_raw"]) if phase in ("joint","predictor") else None,
                "optimizer_state":jax.tree_util.tree_map(np.asarray,state),"states":states_eval})
        final_optimizer_states[phase]=jax.tree_util.tree_map(np.asarray,state)
    variables.update({"generator":generator,"encoder":params["encoder"],"q_raw":jnp.asarray(q_raw),
                      "predictor":predictor,"folded_predictor":folded,"target_states":target_states,
                      "optimizer_states":final_optimizer_states})
    return variables,permutations,arrays|{"history_marker":np.asarray([global_step],np.int64)},pre,True,history


def make_q_diagnostic(config, trust=False):
    cg_max=config["q"] if trust else 2*config["q"]
    def residual(gen,q,affine,truth,coords,mask,mean,scales):
        state=jnp.concatenate((affine,q)); coeff=apply_generator(gen,q[None],mean,scales,config)[0]
        prediction=p4.decode_one_cox_jax(state,coeff,coords,mask,p7.H1)
        return (prediction-truth)/jnp.sqrt(jnp.maximum(jnp.sum(truth**2),1e-300))
    def one_capacity(gen,q,affine,truth,coords,mask,mean,scales):
        fn=lambda value: residual(gen,value,affine,truth,coords,mask,mean,scales)
        r,pull=jax.vjp(fn,q); g=pull(r)[0]; right=-g; rr0=jnp.vdot(right,right)
        def matvec(v):
            jv=jax.jvp(fn,(q,),(v,))[1]
            return pull(jv)[0]+1e-12*v
        def body(carry,_):
            x,r0,d,rr,running,count,breakdown=carry
            def step(values):
                x,r0,d,rr,_,count,breakdown=values; ad=matvec(d); denom=jnp.vdot(d,ad)
                valid=jnp.isfinite(denom)&(denom>0); alpha=jnp.where(valid,rr/denom,0.)
                xn=x+alpha*d; rn=r0-alpha*ad; rrn=jnp.vdot(rn,rn)
                converged=jnp.sqrt(rrn)<=1e-12*jnp.maximum(jnp.sqrt(rr0),1e-300)
                beta=jnp.where(valid&(rr>0),rrn/rr,0.); dn=rn+beta*d
                return xn,rn,dn,rrn,valid&~converged,count+valid.astype(jnp.int32),breakdown|~valid
            return lax.cond(running,step,lambda x:x,carry),None
        init=(jnp.zeros_like(q),right,right,rr0,rr0>0,jnp.int32(0),jnp.bool_(False))
        (delta,_,_,rr,_,iters,breakdown),_=lax.scan(body,init,None,length=cg_max)
        tangent=r+jax.jvp(fn,(q,),(delta,))[1]
        mapping=q-jnp.clip(q-g,-1.,1.)
        gamma=jnp.linalg.norm(mapping)/jnp.maximum(1.,jnp.linalg.norm(q))
        eta=jnp.linalg.norm(tangent)/jnp.maximum(jnp.linalg.norm(r),1e-300)
        return gamma,eta,jnp.vdot(r,r),jnp.vdot(tangent,tangent),delta,iters,jnp.sqrt(rr)/jnp.maximum(jnp.sqrt(rr0),1e-300),breakdown
    return jax.jit(jax.vmap(one_capacity,in_axes=(None,0,0,0,None,None,None,None)))


def capacity_metrics(datasets,generator,q,mean,scales,arrays):
    fn=make_q_diagnostic({**ARMS["T1"],"q":q.shape[1]} if q.shape[1]==19 else ARMS["T2"])
    offset=0; report={"meshes":{}}; pooled={name:[] for name in ("gamma","eta","residual","tangent","bound")}
    for item in datasets:
        count=len(item["flat"]); values=[]
        for start in range(0,count,8):
            take=slice(start,min(start+8,count)); values.append(tuple(map(np.asarray,fn(
                generator,jnp.asarray(q[offset+start:offset+min(start+8,count)]),jnp.asarray(item["affine"][take]),
                jnp.asarray(item["flat"][take]),jnp.asarray(item["coords"]),jnp.asarray(item["mask"]),
                jnp.asarray(mean),jnp.asarray(scales)))))
        merged=[np.concatenate([row[i] for row in values]) for i in range(8)]
        gamma,eta,resid,tangent,delta,iters,cgrel,breakdown=merged
        bound=np.mean(np.minimum(q[offset:offset+count]+1,1-q[offset:offset+count])<=1e-6,axis=1)
        for name,value in zip(("gamma","eta","residual","tangent","delta","cg_iterations","cg_relative","cg_breakdown","bound_fraction"),
                              (gamma,eta,resid,tangent,delta,iters,cgrel,breakdown,bound)):
            arrays[f"capacity_N{item['N']}_{name}"]=value
        report["meshes"][str(item["N"])]=capacity_summary(gamma,eta,resid,tangent,bound,breakdown)
        for name,value in (("gamma",gamma),("eta",eta),("residual",resid),("tangent",tangent),("bound",bound)):
            pooled[name].extend(value.tolist())
        offset += count
    pooled_breakdown=np.concatenate([np.asarray(arrays[f"capacity_N{item['N']}_cg_breakdown"],bool) for item in datasets])
    report["pooled"]=capacity_summary(*(np.asarray(pooled[x]) for x in ("gamma","eta","residual","tangent","bound")),pooled_breakdown)
    return report


def capacity_summary(gamma,eta,residual,tangent,bound,breakdown):
    return {"gamma_median":float(np.quantile(gamma,.5,method="linear")),
            "gamma_p95":float(np.quantile(gamma,.95,method="linear")),
            "eta_median":float(np.quantile(eta,.5,method="linear")),
            "eta_p10":float(np.quantile(eta,.1,method="linear")),
            "eta_E":float(np.sqrt(np.sum(tangent)/max(np.sum(residual),1e-300))),
            "bound_component_fraction":float(np.mean(bound)),
            "cg_breakdown_count":int(np.sum(breakdown)),"all_finite":bool(all(np.all(np.isfinite(x)) for x in (gamma,eta,residual,tangent,bound)))}


def make_trust_attempt(config):
    def residual(gen,q,affine,truth,coords,mask,mean,scales):
        state=jnp.concatenate((affine,q)); coeff=apply_generator(gen,q[None],mean,scales,config)[0]
        pred=p4.decode_one_cox_jax(state,coeff,coords,mask,p7.H1)
        return (pred-truth)/jnp.sqrt(jnp.maximum(jnp.sum(truth**2),1e-300))
    def one(gen,q,affine,truth,coords,mask,mean,scales,delta,damping,active,recorded):
        def perform(_):
            fn=lambda value: residual(gen,value,affine,truth,coords,mask,mean,scales)
            r,pull=jax.vjp(fn,q); g=pull(r)[0]
            def matvec(v): return pull(jax.jvp(fn,(q,),(v,))[1])[0]+damping*v
            right=-g; rr0=jnp.vdot(right,right)
            def body(carry,_):
                x,r0,d,rr,running,count,breakdown=carry
                def step(values):
                    x,r0,d,rr,_,count,breakdown=values; ad=matvec(d); denom=jnp.vdot(d,ad)
                    valid=jnp.isfinite(denom)&(denom>0); alpha=jnp.where(valid,rr/denom,0.)
                    xn=x+alpha*d; rn=r0-alpha*ad; rrn=jnp.vdot(rn,rn); conv=jnp.sqrt(rrn)<=1e-12*jnp.maximum(jnp.sqrt(rr0),1e-300)
                    beta=jnp.where(valid&(rr>0),rrn/rr,0.); return xn,rn,rn+beta*d,rrn,valid&~conv,count+valid.astype(jnp.int32),breakdown|~valid
                return lax.cond(running,step,lambda x:x,carry),None
            init=(jnp.zeros_like(q),right,right,rr0,rr0>0,jnp.int32(0),jnp.bool_(False))
            (step,_,_,rr,_,iters,breakdown),_=lax.scan(body,init,None,length=config["q"])
            step*=jnp.minimum(1.,delta/jnp.maximum(jnp.linalg.norm(step),1e-300)); trial=jnp.clip(q+step,-1,1); actual_step=trial-q
            jstep=jax.jvp(fn,(q,),(actual_step,))[1]; predicted=-jnp.vdot(g,actual_step)-.5*jnp.vdot(jstep,jstep)
            trial_r=fn(trial); objective=jnp.vdot(r,r); trial_objective=jnp.vdot(trial_r,trial_r); actual=.5*(objective-trial_objective)
            rho_defined=predicted>0; rho=jnp.where(rho_defined,actual/predicted,0.)
            finite=jnp.all(jnp.isfinite(trial_r))&jnp.isfinite(predicted)&jnp.isfinite(actual)&jnp.isfinite(rho)
            accepted=(~breakdown)&finite&rho_defined&(actual>0)&(rho>=ACCEPT_RHO)
            terminate=accepted&(((objective-trial_objective)/jnp.maximum(objective,1e-300)<=1e-12)|(jnp.linalg.norm(actual_step)/(1+jnp.linalg.norm(q))<=1e-12))
            shrink=(~accepted)|((rho_defined)&(rho<.25)); expand=accepted&(rho>.75)&(jnp.linalg.norm(actual_step)>=.9*delta); improve=accepted&(rho>.75)
            nd=jnp.where(shrink,jnp.maximum(delta/4,DELTA_MIN),jnp.where(expand,jnp.minimum(2*delta,DELTA_MAX),delta))
            nl=jnp.where(shrink,jnp.minimum(10*damping,LAMBDA_MAX),jnp.where(improve,jnp.maximum(damping/3,LAMBDA_MIN),damping))
            return (jnp.where(accepted,trial,q),jnp.where(accepted,trial_objective,objective),nd,nl,~terminate,
                    objective,trial_objective,predicted,actual,rho,rho_defined,accepted,terminate,iters,
                    jnp.sqrt(rr)/jnp.maximum(jnp.sqrt(rr0),1e-300),breakdown,finite,actual_step,g)
        def inactive(_):
            z=jnp.asarray(0.,F64); return q,recorded,delta,damping,jnp.bool_(False),recorded,recorded,z,z,z,jnp.bool_(False),jnp.bool_(False),jnp.bool_(False),jnp.int32(0),z,jnp.bool_(False),jnp.bool_(True),jnp.zeros_like(q),jnp.zeros_like(q)
        return lax.cond(active,perform,inactive,None)
    return jax.jit(jax.vmap(one,in_axes=(None,0,0,0,None,None,None,None,0,0,0,0)))


def run_trust(datasets,generator,starts,mean,scales,config,arrays,args,smoke):
    attempts=1 if smoke else TRUST_ATTEMPTS; total=starts.shape[1]; qdim=config["q"]
    trace={"q":np.empty((2,total,attempts+1,qdim)),"objective":np.empty((2,total,attempts+1)),
           "delta":np.empty((2,total,attempts+1)),"damping":np.empty((2,total,attempts+1)),
           "active":np.zeros((2,total,attempts+1),bool),
           "trial_objective":np.empty((2,total,attempts)),"predicted":np.empty((2,total,attempts)),
           "actual":np.empty((2,total,attempts)),"rho":np.empty((2,total,attempts)),
           "rho_defined":np.zeros((2,total,attempts),bool),"accepted":np.zeros((2,total,attempts),bool),
           "terminated":np.zeros((2,total,attempts),bool),"attempted":np.zeros((2,total,attempts),bool),
           "cg_iterations":np.zeros((2,total,attempts),np.int32),"cg_relative":np.empty((2,total,attempts)),
           "cg_breakdown":np.zeros((2,total,attempts),bool),"finite":np.zeros((2,total,attempts),bool),
           "step":np.empty((2,total,attempts,qdim)),"gradient":np.empty((2,total,attempts,qdim)),
           "jvp_count":np.zeros((2,total,attempts),np.int32),"vjp_count":np.zeros((2,total,attempts),np.int32)}
    fn=make_trust_attempt(config)
    for start_index in range(2):
        q=starts[start_index].copy(); delta=np.full(total,DELTA0); damping=np.full(total,LAMBDA0); active=np.ones(total,bool)
        objective=np.empty(total); off=0
        for item in datasets:
            count=len(item["flat"]); state=np.concatenate((item["affine"],q[off:off+count]),axis=1)
            temp={}; evaluate_full([item],generator,state,mean,scales,config,temp,"initial",False)
            objective[off:off+count]=temp[f"initial_N{item['N']}_numerator"]/temp[f"initial_N{item['N']}_denominator"]; off+=count
        trace["q"][start_index,:,0]=q; trace["objective"][start_index,:,0]=objective
        trace["delta"][start_index,:,0]=delta; trace["damping"][start_index,:,0]=damping; trace["active"][start_index,:,0]=active
        for attempt in range(attempts):
            attempted=active.copy(); current_delta=delta.copy(); current_damping=damping.copy(); off=0
            for item in datasets:
                count=len(item["flat"]); sl=slice(off,off+count)
                result=tuple(map(np.asarray,fn(generator,jnp.asarray(q[sl]),jnp.asarray(item["affine"]),jnp.asarray(item["flat"]),jnp.asarray(item["coords"]),jnp.asarray(item["mask"]),jnp.asarray(mean),jnp.asarray(scales),jnp.asarray(delta[sl]),jnp.asarray(damping[sl]),jnp.asarray(active[sl]),jnp.asarray(objective[sl]))))
                (nq,no,nd,nl,na,old,trial,pred,actual,rho,rdef,accept,term,iters,cgrel,breakdown,finite,step,grad)=result
                for name,value in (("trial_objective",trial),("predicted",pred),("actual",actual),("rho",rho),("rho_defined",rdef),("accepted",accept),("terminated",term),("cg_iterations",iters),("cg_relative",cgrel),("cg_breakdown",breakdown),("finite",finite),("step",step),("gradient",grad)):
                    trace[name][start_index,sl,attempt]=value
                q[sl],objective[sl],delta[sl],damping[sl],active[sl]=nq,no,nd,nl,na; off+=count
            trace["attempted"][start_index,:,attempt]=attempted
            if not (np.array_equal(current_delta,trace["delta"][start_index,:,attempt])
                    and np.array_equal(current_damping,trace["damping"][start_index,:,attempt])):
                raise SystemExit("trust state transition persistence mismatch")
            trace["jvp_count"][start_index,:,attempt]=np.where(attempted,trace["cg_iterations"][start_index,:,attempt]+1,0)
            trace["vjp_count"][start_index,:,attempt]=np.where(attempted,trace["cg_iterations"][start_index,:,attempt]+1,0)
            trace["q"][start_index,:,attempt+1]=q; trace["objective"][start_index,:,attempt+1]=objective
            trace["delta"][start_index,:,attempt+1]=delta; trace["damping"][start_index,:,attempt+1]=damping; trace["active"][start_index,:,attempt+1]=active
            atomic_json(args.progress_json,{"status":"in_progress","phase":"trust","start":start_index,"attempt":attempt+1,"scientific_metrics_exposed":False})
    for name,value in trace.items(): arrays["trust_"+name]=value
    arrays["trust_unhealthy_exhaustion"]=(trace["active"][:,:,-1]
        & (trace["delta"][:,:,-1]<=DELTA_MIN) & (trace["damping"][:,:,-1]>=LAMBDA_MAX))
    return trace


def selection_coefficients(args,selection,smoke):
    if smoke:
        return np.zeros((sum(len(x["flat"]) for x in selection),3328),np.float64)
    with np.load(args.p4_npz,allow_pickle=False) as data:
        return np.concatenate([np.asarray(data[f"H1_N{x['N']}_coefficients"],np.float64).reshape(-1,3328)
                               for x in selection])


def t2_structural_preflight(config,mean,scales,smoke=False):
    """Actual q32 Cox-weak/K3-full route paired with a same-job live FOM."""
    n,steps,cases=(32,1,1) if smoke else (1024,50,4)
    if smoke:
        parameters={"cx":np.asarray((.42,)),"cy":np.asarray((.58,)),"width":np.asarray((.12,)),
                    "amplitude":np.asarray((1.2,)),"nu":np.asarray((.01,))}
        truth=np.asarray(base.c.bf.blob_ic(n,.42,.58,.12,1.2))[None,None,:]; dummy=None
    else:
        truth,parameters,_,dummy=base.generate_live_reference(n)
    feature_rows=[]
    for case in range(cases):
        one={k:parameters[k][case:case+1] for k in ("cx","cy","width","amplitude","nu")}
        feature_rows.append(c.trajectory_features(one,n)[0,:steps+1])
    geometry=base.weak_geometry(n,p7.H1); stencil=jnp.asarray(geometry["stencil_coords"],F64)
    stencil_mask=jnp.asarray(geometry["stencil_mask"],F64); phi=jnp.asarray(geometry["phi_weighted"],F64); eigen=jnp.asarray(geometry["eigenvalues"],F64)
    coords=jnp.asarray(c.grid_coords(n),F64); mask=jnp.asarray(c.binary_boundary_mask(n),F64)
    coarse=jnp.asarray(p3.span_polynomial_table_np(48),F64); fine=jnp.asarray(p3.span_polynomial_table_np(32),F64)
    decoder=p4.make_pallas_hierarchical_decoder(steps+1,n*n,p7.H1)
    generator=init_generator(config,20260826); predictor=init_predictor(config,20260827)
    def route(pred,gen,features,nu,mean_arg,scales_arg):
        states=apply_predictor(pred,features); coeff=apply_generator(gen,states[:,5:],mean_arg,scales_arg,config)
        def weak(st,a,prev,pa):
            current=p4.decode_one_cox_jax(st,a,stencil,stencil_mask,p7.H1).reshape(p7.H1["m"],5)
            previous=p4.decode_one_cox_jax(prev,pa,stencil[::5],stencil_mask[::5],p7.H1)
            center,xp,xm,yp,ym=[current[:,i] for i in range(5)]; dx=1/(n-1)
            adv=center*(jnp.where(center>0,(center-xm)/dx,(xp-center)/dx)+jnp.where(center>0,(center-ym)/dx,(yp-center)/dx))
            projected=phi.T@center; residual=(1+c.DT*nu*eigen)**-1*(phi.T@(center-previous)+c.DT*(phi.T@adv+nu*eigen*projected))
            return residual
        residual=jax.vmap(weak)(states[1:],coeff[1:],states[:-1],coeff[:-1])
        fields=decoder(states,coeff,coords,mask,coarse,fine)
        control=p4.decode_states_cox_sequential(states,coeff,coords,mask,p7.H1)
        return fields,control,residual,states,coeff
    route_jit=jax.jit(route); lowered=route_jit.lower(predictor,generator,jnp.asarray(feature_rows[0]),jnp.asarray(parameters["nu"][0]),jnp.asarray(mean),jnp.asarray(scales)); executable=lowered.compile()
    memory=base.memory_analysis(executable)["eligibility_device_bytes"]
    route_args=[(predictor,generator,jnp.asarray(feature_rows[i]),jnp.asarray(parameters["nu"][i]),jnp.asarray(mean),jnp.asarray(scales)) for i in range(cases)]
    if smoke:
        fields,control,residual,states,coeff=map(np.asarray,executable(*route_args[0]))
        identity={"relative_l2":float(np.linalg.norm(fields-control)/max(np.linalg.norm(control),1e-300)),
            "boundary":bool(np.all(fields[:,np.asarray(mask)==0]==0)),"finite":bool(all(np.all(np.isfinite(x)) for x in (fields,control,residual,states,coeff)))}
        return {"pass":False,"scientific":False,"same_invocation_identity":identity,"compiled_device_bytes":int(memory)}
    fom=base.bc.make_chain(n,base.FOM_OUTER,lin_tol=base.FOM_INNER,preconditioner="helmholtz")[0]
    def invoke(method,case,return_output=False):
        started=time.perf_counter()
        work={}
        if method=="fom": out=fom(jnp.asarray(truth[case,0]),parameters["nu"][case],dummy,jnp.int32(5))
        else:
            recovered,sample_indices=c.recover_blob_parameters_fixed_sample(truth[case,0],n)
            one={"cx":recovered[0:1],"cy":recovered[1:2],"width":recovered[2:3],"amplitude":recovered[3:4],"nu":parameters["nu"][case:case+1]}
            args=list(route_args[case]); args[2]=jnp.asarray(c.trajectory_features(one,n)[0,:steps+1]); out=executable(*tuple(args))
            work={"cold_sample_count":int(sample_indices.size),"cold_recovery_finite":bool(np.all(np.isfinite(recovered)))}
        jax.block_until_ready(out); elapsed=float(time.perf_counter()-started)
        return (out,elapsed,work) if return_output else elapsed
    for method in ("fom","rom"): invoke(method,0)
    burn_count=c.gpu_burn(3.0); records={"fom":[],"rom":[]}; orders=[]
    for rep in range(24):
        order=("fom","rom") if rep%2==0 else ("rom","fom"); orders.append(order)
        for case in range(cases):
            for position,method in enumerate(order):
                output,elapsed,invoke_work=invoke(method,case,True); row={"repetition":rep,"case_index":case,"position":position,"elapsed_s":elapsed,**invoke_work}
                if method=="fom": row.update(base.fom_grade(output,truth[case]))
                else:
                    fields,control,residual,states,coeff=map(np.asarray,output)
                    row.update({"identity_relative_l2":float(np.linalg.norm(fields-control)/max(np.linalg.norm(control),1e-300)),
                        "exact_boundary":bool(np.all(fields[:,np.asarray(mask)==0]==0)),
                        "finite":bool(all(np.all(np.isfinite(x)) for x in (fields,control,residual,states,coeff))),
                        "cold_sample_count":invoke_work["cold_sample_count"],"cold_recovery_finite":invoke_work["cold_recovery_finite"],
                        "cox_weak_evaluations":int(residual.shape[0]),"k3_coefficient_grid_full_field_evaluations":int(fields.shape[0]),
                        "weak_jacobian_evaluations":0,"trial_evaluations":0,"failures":0})
                records[method].append(row)
    summaries={method:base.summarize_timing(rows,cases) for method,rows in records.items()}
    per_case={method:summaries[method]["per_case_median_elapsed_s"] for method in records}
    speed=float(np.median(per_case["fom"])/np.median(per_case["rom"])); ci=base.clustered_speedup_ci(per_case["fom"],per_case["rom"],20266100)
    fom_eligible=bool(all(row["finite"] and row["breakdowns"]==0 and row["flags_nonzero"]==0 and row["max_returned_relative_residual"]<=base.FOM_OUTER for row in records["fom"])
        and np.mean([row["trajectory_relative_l2"] for row in records["fom"]])<=1e-3 and np.max([row["trajectory_relative_l2"] for row in records["fom"]])<=3e-3)
    identity_pass=all(x["identity_relative_l2"]<=IDENTITY_TOL and x["exact_boundary"] and x["finite"]
        and x["cox_weak_evaluations"]==50 and x["k3_coefficient_grid_full_field_evaluations"]==51
        and x["weak_jacobian_evaluations"]==0 and x["trial_evaluations"]==0 and x["failures"]==0 for x in records["rom"])
    gate=bool(fom_eligible and identity_pass and memory<=20_000_000_000 and speed>=10 and ci[0]>=8)
    return {"scientific":True,"compiled_device_bytes":int(memory),"memory_pass":memory<=20_000_000_000,
            "work":{"cox_weak_evaluations":50,"k3_coefficient_grid_full_field_evaluations":51,"weak_jacobian_evaluations":0,"trial_evaluations":0,"failures":0},
            "fom_eligible":fom_eligible,"burn_count":int(burn_count),"orders":[list(x) for x in orders],"position_counts":{"fom":[12,12],"rom":[12,12]},"records":records,
            "summaries":summaries,"per_case_median_seconds":per_case,"paired_median_speedup":speed,"clustered_speedup_ci":ci,"pass":gate}


def license_capacity(train_metrics,capacity,history):
    failing=[]
    for key,row in list(train_metrics["meshes"].items())+[("pooled",train_metrics["pooled"])]:
        if not (row["trajectory_error_mean"]<=2e-4 and row["trajectory_error_worst"]<=7e-4):
            failing.append(key)
    def loss(epoch,key):
        metrics=history[f"joint_epoch{epoch}"]["metrics"]
        return metrics["pooled"]["mean_snapshot_relative_l2_squared"] if key=="pooled" else metrics["meshes"][key]["mean_snapshot_relative_l2_squared"]
    improvement={key:(loss(24,key)-loss(27,key))/max(loss(24,key),1e-300) for key in [*train_metrics["meshes"],"pooled"]}
    def clauses(key):
        row=capacity[key] if key=="pooled" else capacity["meshes"][key]
        return {"late_improvement":bool(0<=improvement[key]<=.01),
                "gamma":bool(row["gamma_median"]<=1e-4 and row["gamma_p95"]<=1e-3),
                "eta":bool(row["eta_median"]>=.9 and row["eta_p10"]>=.8 and row["eta_E"]>=.8),
                "bound":bool(row["bound_component_fraction"]<=.01),
                "finite":bool(row["all_finite"] and row["cg_breakdown_count"]==0)}
    tested={key:clauses(key) for key in [*train_metrics["meshes"],"pooled"]}
    applicable=sorted(set(failing+["pooled"]))
    licensed=bool(failing and all(all(tested[key].values()) for key in applicable))
    return {"failing_strata":failing,"applicable_strata":applicable,"late_improvement_24_27":improvement,
            "criteria":tested,"g2_licensed":licensed}


def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--arm",choices=("T1","T2"),required=True)
    for phase in ("p4","p5","p6"):
        for kind in ("json","npz","audit","manifest"):
            parser.add_argument(f"--{phase}-{kind}",dest=f"{phase}_{kind}")
    for kind in ("json","npz","checkpoint","audit","manifest"):
        parser.add_argument(f"--p7-{kind}",dest=f"p7_{kind}")
    for kind in ("json","npz","audit","manifest"):
        parser.add_argument(f"--p8-{kind}",dest=f"p8_{kind}")
    for kind in ("json","npz","checkpoint","audit","manifest"):
        parser.add_argument(f"--t1-{kind}",dest=f"t1_{kind}")
    parser.add_argument("--target-dir"); parser.add_argument("--prereg")
    parser.add_argument("--output-json",required=True); parser.add_argument("--output-npz",required=True)
    parser.add_argument("--checkpoint",required=True); parser.add_argument("--progress-json",required=True)
    parser.add_argument("--work-checkpoint",required=True); parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--smoke-skip-structural",action="store_true")
    return parser.parse_args()


def main():
    args=parse_args(); c.require_gpu_highest(); started=time.perf_counter(); config=dict(ARMS[args.arm]); smoke=args.smoke
    if smoke:
        reports=bindings=None
    else:
        required=[getattr(args,f"{phase}_{kind}") for phase in ("p4","p5","p6") for kind in ("json","npz","audit","manifest")]
        required += [getattr(args,f"p7_{kind}") for kind in ("json","npz","checkpoint","audit","manifest")]
        required += [getattr(args,f"p8_{kind}") for kind in ("json","npz","audit","manifest")]+[args.target_dir,args.prereg]
        if any(x is None for x in required): raise SystemExit("scientific Phase9 requires exact P4-P8 chain")
        reports,bindings=validate_chains(args,args.arm)
    train,selection,coefficients,target_records=load_populations(args,smoke)
    features=legacy.concatenate(train,"features"); norm=train_normalization(coefficients,features)
    views=per_n_views(train,norm["normalized"])
    if args.smoke_skip_structural and not (smoke and args.arm=="T2"):
        raise SystemExit("--smoke-skip-structural is T2 synthetic-only")
    structural=(t2_structural_preflight(config,norm["mean"],norm["scales"],smoke)
                if args.arm=="T2" and not args.smoke_skip_structural else
                {"scientific":False,"separately_smoked":True,"pass":False} if args.arm=="T2" else None)
    if structural is not None and not structural["pass"] and not smoke:
        initial_q=np.zeros((len(coefficients),config["q"]),np.float64)
        variables={"generator":init_generator(config),"encoder":init_encoder(config),"predictor":init_predictor(config),
                   "q_raw":jnp.asarray(initial_q),"target_states":np.concatenate((legacy.concatenate(train,"affine"),initial_q),axis=1)}
        train_result=(variables,{}, {},{"structural_preflight":structural,"weights_bitwise_unchanged":True},False)
    else:
        train_result=train_all(config,train,views,norm,args,smoke,started)
    if len(train_result)==5:
        variables,permutations,arrays,preflight,updates_started=train_result; history={}
    else:
        variables,permutations,arrays,preflight,updates_started,history=train_result
    arrays.update({"coefficient_mean":norm["mean"],"head_scales":norm["scales"],
                   "predictor_feature_mean":norm["feature_mean"],"predictor_feature_scale":norm["feature_scale"],
                   "predictor_feature_empirical_scale":norm["feature_empirical"],
                   "training_features":features,"normalization_source_indices":norm["source_indices"]})
    arrays.update(permutations)
    parameter_counts={"generator":tree_count(variables["generator"]),"encoder":tree_count(variables["encoder"]),
                      "predictor":tree_count(variables["predictor"])}
    expected_counts={key:config[key+"_count"] for key in ("generator","encoder","predictor")}
    if parameter_counts != expected_counts: raise SystemExit("arm parameter count mismatch")
    decision={"arm":args.arm,"updates_started":updates_started,"infrastructure_preflight_pass":bool(updates_started),
              "train_pass":False,"selection_evaluated":False,"phase9_pass":False,"g2_licensed":False,
              "scientific_promotion_allowed":False,"next_action":"hard stop"}
    terminal_train=capacity=capacity_license=selection_report=trust_health=None
    predictor_fold_identity=None
    if updates_started:
        arrays["final_q_raw"]=np.asarray(variables["q_raw"])
        arrays["final_target_states"]=np.asarray(variables["target_states"])
        arrays["update_resolution_order"]=(np.asarray((16,16,16),np.int16) if smoke else
            np.tile(np.asarray(N_ORDER,np.int16),sum(PHASES[x]["epochs"] for x in PHASES)*BATCHES_PER_N))
        predictor_fold_identity=legacy.predictor_fold_identity(variables["predictor"],variables["folded_predictor"],
            legacy.concatenate(train,"features"),norm["feature_mean"],norm["feature_scale"])
        q=np.tanh(np.asarray(variables["q_raw"])); train_states=np.concatenate((legacy.concatenate(train,"affine"),q),axis=1)
        terminal_train=evaluate_full(train,variables["generator"],train_states,norm["mean"],norm["scales"],config,arrays,"terminal_train",not smoke)
        train_health=bool(all(row["all_finite"] and row["boundary_violation_count"]==0 and row["k3_cox_identity_worst"]<=IDENTITY_TOL
                              for row in list(terminal_train["meshes"].values())+[terminal_train["pooled"]])) if not smoke else True
        train_pass=gate(terminal_train,2e-4,7e-4,not smoke); decision["train_pass"]=train_pass
        if args.arm=="T1" and train_health and not train_pass and not smoke:
            capacity=capacity_metrics(train,variables["generator"],q,norm["mean"],norm["scales"],arrays)
            capacity_license=license_capacity(terminal_train,capacity,history)
            decision["g2_licensed"]=capacity_license["g2_licensed"]
        if train_pass:
            selection_coeff=selection_coefficients(args,selection,smoke)
            normalized_selection=np.concatenate(((selection_coeff[:,:2304]-norm["mean"][:2304])/norm["scales"][0],
                                                  (selection_coeff[:,2304:]-norm["mean"][2304:])/norm["scales"][1]),axis=1)
            encoder_raw=encode_batches(variables["encoder"],normalized_selection,config); encoder_q=np.tanh(encoder_raw)
            direct_states=predictor_batches(variables["folded_predictor"],legacy.concatenate(selection,"features")); predictor_q=direct_states[:,5:]
            arrays["selection_free_encoder_q_raw"]=encoder_raw; arrays["selection_direct_states"]=direct_states
            trace=run_trust(selection,variables["generator"],np.stack((predictor_q,encoder_q)),norm["mean"],norm["scales"],config,arrays,args,smoke)
            best=np.argmin(trace["objective"][:,:,-1],axis=0); bestq=np.where(best[:,None]==0,trace["q"][0,:,-1],trace["q"][1,:,-1])
            arrays["trust_chosen_start"]=best.astype(np.int8); arrays["trust_chosen_terminal_q"]=bestq
            oracle_states=np.concatenate((legacy.concatenate(selection,"affine"),bestq),axis=1)
            direct=evaluate_full(selection,variables["generator"],direct_states,norm["mean"],norm["scales"],config,arrays,"selection_direct",True)
            oracle=evaluate_full(selection,variables["generator"],oracle_states,norm["mean"],norm["scales"],config,arrays,"selection_oracle",True)
            ratios={key:(direct["pooled"] if key=="pooled" else direct["meshes"][key])["trajectory_error_mean"]/
                    max((oracle["pooled"] if key=="pooled" else oracle["meshes"][key])["trajectory_error_mean"],1e-300)
                    for key in [*direct["meshes"],"pooled"]}
            defined=trace["rho_defined"]&trace["attempted"]; undefined=(~trace["rho_defined"])&trace["attempted"]
            breakdown_count=int(np.sum(trace["cg_breakdown"]&trace["attempted"])); exhaustion_count=int(np.sum(arrays["trust_unhealthy_exhaustion"]))
            trust_health={"finite":bool(np.all(trace["finite"]|~trace["attempted"])),"breakdown_count":breakdown_count,
                          "no_breakdown":breakdown_count==0,"unhealthy_exhaustion_count":exhaustion_count,"no_unhealthy_exhaustion":exhaustion_count==0,
                          "defined_rho_match":bool(np.allclose(trace["rho"][defined],trace["actual"][defined]/trace["predicted"][defined],rtol=2e-13,atol=2e-14)),
                          "undefined_rho_zero":bool(np.all(trace["rho"][undefined]==0.0)),"undefined_never_accepted":bool(not np.any(trace["accepted"][undefined]))}
            trust_gate=trust_health_gate(trust_health)
            selection_pass=bool(gate(direct,3e-4,1e-3,True) and gate(oracle,2e-4,7e-4,True)
                                and all(x<=1.5 for x in ratios.values()) and trust_gate)
            selection_report={"direct":direct,"oracle":oracle,"direct_oracle_mean_ratio":ratios,"trust_health":trust_health,"trust_gate":trust_gate,"pass":selection_pass}
            decision.update({"selection_evaluated":True,"phase9_pass":selection_pass,"scientific_promotion_allowed":selection_pass,
                             "next_action":"separate corrected-rollout proposal" if selection_pass else "hard stop"})
        elif decision["g2_licensed"]:
            decision["next_action"]="conditional P9-T2 root audit"
    os.makedirs(os.path.dirname(os.path.abspath(args.output_npz)),exist_ok=True)
    np.savez_compressed(args.output_npz,**arrays)
    checkpoint={"status":"excluded_execution_smoke" if smoke else "complete","arm":args.arm,"config":config,
                "generator":jax.tree_util.tree_map(np.asarray,variables["generator"]),"encoder":jax.tree_util.tree_map(np.asarray,variables["encoder"]),
                "predictor":jax.tree_util.tree_map(np.asarray,variables["predictor"]),"q_raw":np.asarray(variables["q_raw"]),
                "folded_predictor":jax.tree_util.tree_map(np.asarray,variables.get("folded_predictor",variables["predictor"])),
                "encoder_handoff_q_raw":np.asarray(arrays.get("encoder_handoff_q_raw",np.empty((0,config["q"])))),
                "final_target_states":np.asarray(variables["target_states"]),
                "optimizer_states":jax.tree_util.tree_map(np.asarray,variables.get("optimizer_states",{})),
                "normalization":{k:norm[k] for k in ("mean","scales","feature_mean","feature_scale")}}
    atomic_pickle(args.checkpoint,checkpoint)
    report={"status":"excluded_execution_smoke" if smoke else "complete","provenance":c.provenance(),"arm":config,
            "bindings":bindings,"parameter_counts":{"reported":parameter_counts,"expected":expected_counts},
            "data":{"training":legacy.metadata(train),"selection":legacy.metadata(selection),
                    "target_chunks":target_records,"train_snapshot_count":int(len(coefficients)),
                    "selection_snapshot_count":int(sum(len(x["flat"]) for x in selection))},
            "information_boundary":{"train_only_weights":True,"selection_target_coefficients_training_use":False,
                "model_validation_touched":False,"confirmation_touched":False,"weak_eq_touched":False,"scaling_touched":False},
            "normalization":{"definition":"train-only vector mean and centered per-head RMS","elapsed_s":norm["elapsed_s"],
                "host_bytes":norm["host_bytes"],"mean_sha256":array_sha(norm["mean"]),"scales_sha256":array_sha(norm["scales"]),
                "normalized_training_coefficients_sha256":array_sha(norm["normalized"]),
                "training_features_sha256":array_sha(features),"feature_mean_sha256":array_sha(norm["feature_mean"]),
                "feature_scale_sha256":array_sha(norm["feature_scale"]),"feature_empirical_scale_sha256":array_sha(norm["feature_empirical"])},
            "schedule":{"phase_order":["encoder","joint","predictor"],"epochs":{k:(1 if smoke else v["epochs"]) for k,v in PHASES.items()},
                "resolution_cycle":[64,128,256],"batches":{"64":8,"128":2,"256":1},"batches_per_N_epoch":1 if smoke else 3264,
                "phase_update_counts":{k:(1 if smoke else v["epochs"]*9792) for k,v in PHASES.items()},
                "total_update_count":3 if smoke else 528768,"terminal_epochs":{"encoder":9,"joint":27,"predictor":18},"terminal_only":True},
            "preflight":preflight,"history":history,"terminal_train":terminal_train,"capacity":capacity,
            "t2_structural_preflight":structural,
            "predictor_fold_identity":predictor_fold_identity,
            "capacity_license":capacity_license,"selection":selection_report,"decision":decision,
            "npz":{"basename":os.path.basename(args.output_npz),"sha256":c.sha256(args.output_npz)},
            "checkpoint":{"basename":os.path.basename(args.checkpoint),"sha256":c.sha256(args.checkpoint)},
            "work_checkpoint":({"basename":os.path.basename(args.work_checkpoint),"sha256":c.sha256(args.work_checkpoint)}
                               if os.path.isfile(args.work_checkpoint) else None),
            "elapsed_s":float(time.perf_counter()-started)}
    atomic_json(args.output_json,report); atomic_json(args.progress_json,{"status":"complete","scientific_metrics_exposed":False})
    print(json.dumps({"status":report["status"],"decision":decision},sort_keys=True),flush=True); print("ALL-DONE",flush=True)


if __name__=="__main__":
    main()
