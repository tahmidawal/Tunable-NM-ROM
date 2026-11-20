# Validate scientific-only observer after the same tiny seven native stages.
function b8_progress_checks(out)
    dir=joinpath(out,"tiny_native_model")
    rows=readlines(joinpath(dir,"epoch_progress.tsv"))
    @assert length(rows)==8
    expected=["progress_s$(lpad(s,2,'0'))_e001.jld2" for s in 1:7]
    found=sort(filter(name->startswith(name,"progress_") && endswith(name,".jld2"),readdir(dir)))
    @assert found==expected
    for stage in 1:7
        columns=split(rows[stage+1],'\t';keepempty=true)
        @assert length(columns)==13
        @assert parse(Int,columns[1])==stage && parse(Int,columns[2])==1
        @assert parse(Int,columns[3])==1 && parse(Int,columns[4])==stage && parse(Int,columns[5])==1
        @assert all(v->isfinite(v) && v>=0,parse.(Float64,columns[6:9]))
        @assert all(isfinite,parse.(Float64,columns[10:11]))
        @assert columns[13]==expected[stage]
        saved=JLD2.load(joinpath(dir,expected[stage]))
        progress=saved["progress"]
        @assert progress.stage==stage && progress.epoch==1 && progress.native_training_steps==stage
        @assert progress.model_only_warm_start && !progress.exact_optimizer_resume
        @assert !progress.optimizer_state_serialized && !progress.rng_state_serialized
        @assert haskey(saved,"STATS") && haskey(saved,"metadata")
        for key in ("model","best_model")
            model=saved[key]
            @assert model[1].layers.decoder isa B8BoundaryDecoder
            @assert eltype(model[2])===Float64 && all(isfinite,getdata(model[2]))
            # Loading alone is insufficient: require checkpoint to execute.
            input=(zeros(Float64,2,2),zeros(Float64,2,2))
            prediction,_=model[1](input,model[2],model[3])
            @assert size(prediction)==(1,2) && all(isfinite,prediction)
        end
    end
    Dict("scientific_progress_passed"=>true,"progress_epoch_rows"=>7,
        "progress_snapshots_loaded_and_executed"=>7,"model_only_metadata_verified"=>true)
end
