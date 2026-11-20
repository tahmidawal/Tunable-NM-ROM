# Scientific-only observability. Called AFTER native callback/minconfig update.
# Never mutate optimizer, stopping state, parameters, model state, data or RNG.
function b8_scientific_progress(context, NN, p, st, minconfig, epoch, nepoch,
    num_batches, batch_loop_seconds, callback_seconds, epoch_seconds, loss, stopping)
    isnothing(context) && return nothing
    dir=context.dir; mkpath(dir)
    save_now=epoch==1 || epoch%5==0 || epoch==nepoch || stopping
    checkpoint_seconds=0.0
    checkpoint_file=""
    if save_now
        begin_checkpoint=time_ns()
        ph,sh=Lux.cpu_device()((p,st))
        bph,bsh=Lux.cpu_device()((minconfig.p,minconfig.st))
        model=(NN,ph,sh); best_model=(NN,bph,bsh)
        @assert eltype(ph)===Float64 && eltype(bph)===Float64
        progress=(;stage=context.stage,epoch,requested_stage_epochs=nepoch,
            native_training_steps=B8_FLOAT64_AUDIT["steps"],
            current_development_regularized_loss=Float64(loss),
            best_development_regularized_loss=Float64(minconfig.l),
            native_early_stop_count=minconfig.count,native_patience=minconfig.patience,
            native_stop_triggered=stopping,
            model_only_warm_start=true,exact_optimizer_resume=false,
            optimizer_state_serialized=false,rng_state_serialized=false,
            timing_scope="Host wall time through existing native synchronization; no extra benchmark workloads")
        checkpoint_file="progress_s$(lpad(context.stage,2,'0'))_e$(lpad(epoch,3,'0')).jld2"
        target=joinpath(dir,checkpoint_file);temporary=target*".tmp"
        jldsave(temporary;model,best_model,metadata=context.metadata,
            STATS=deepcopy(context.stats),progress)
        mv(temporary,target;force=true)
        checkpoint_seconds=(time_ns()-begin_checkpoint)/1e9
    end
    path=joinpath(dir,"epoch_progress.tsv")
    new_file=!isfile(path)
    open(path,"a") do io
        new_file && println(io,"stage\tepoch\trequested_stage_epochs\tnative_updates\tbatches\tbatch_loop_wall_seconds\tcallback_and_selection_wall_seconds\tepoch_before_retention_wall_seconds\tcheckpoint_wall_seconds\tdevelopment_regularized_loss\tbest_development_regularized_loss\tnative_stop\tcheckpoint")
        println(io,join((context.stage,epoch,nepoch,B8_FLOAT64_AUDIT["steps"],num_batches,
            batch_loop_seconds,callback_seconds,epoch_seconds,checkpoint_seconds,
            Float64(loss),Float64(minconfig.l),stopping,checkpoint_file),'\t'))
    end
    nothing
end
