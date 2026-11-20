# Native pinned SNF training with matched scalar unit-box Burgers data.
# Physical descriptors are OFFLINE hypernetwork inputs, never online privilege.
using NeuralROMs, CUDA, Lux, LuxCUDA, ComponentArrays, Random, LinearAlgebra
using JLD2, TOML, Statistics
@assert CUDA.functional()
CUDA.allowscalar(false)
@assert ENDIAN_BOM == 0x04030201 "Raw adapter requires little-endian host"
out = ENV["B8_SNF_OUTPUT"]
@assert realpath(pkgdir(NeuralROMs)) == realpath(ENV["B8_SNF_PROJECT"])
include(joinpath(pkgdir(NeuralROMs), "experiments_SNFROM", "smoothNF.jl"))
BLAS.set_num_threads(4)

function readraw(name, shape)
    path = joinpath(out, name * ".f64")
    @assert filesize(path) == 8 * prod(shape)
    values = Vector{Float64}(undef, prod(shape))
    open(path, "r") do io
        read!(io, values)
        @assert eof(io)
    end
    @assert all(isfinite, values)
    reshape(values, Tuple(shape))
end

function prepare_native_data()
    spec = TOML.parsefile(joinpath(out, "raw-data.toml"))
    @assert spec["dtype"] == "Float64" && spec["byte_order"] == "little" && spec["order"] == "F"
    @assert spec["train_count"] == 160 && spec["dev_count"] == 52
    @assert spec["x_shape"] == [2, 16384]
    @assert spec["u_shape"] == [1, 16384, 212, 11]
    @assert spec["t_shape"] == [11] && spec["mu_shape"] == [16, 212]
    x = readraw("x", spec["x_shape"])
    u = readraw("u", spec["u_shape"])
    t = readraw("t", spec["t_shape"])
    physical = readraw("mu", spec["mu_shape"])
    @assert maximum(abs.(mean(physical[:, 1:160]; dims=2))) < 1e-10
    mu = [copy(physical[:, i]) for i in axes(physical, 2)]
    metadata = (; grid=(128,128), role="160 training / 52 development trajectories, no final test",
        descriptor_schema="background, nu, two canonical (A,cx,cy,sigma_t,sigma_n,sin2theta,cos2theta) slots",
        descriptor_statistics_file="descriptor_statistics.json",
        descriptor_normalization="16 physical features standardized from training trajectories only; native time normalization training-only",
        online_descriptor_use=false, native_source_port="qualified isolated Float64 training port",
        matched_sources_file="matched_sources.json")
    file = joinpath(out, "matched_native_data.jld2")
    jldsave(file; x, u, t, mu, metadata)
    file
end

function main()
    file = prepare_native_data()
    GC.gc() # Drop bridge copies before native makedata expands point observations.
    split = (; Ix=:, _Ib=collect(1:160), Ib_=collect(161:212), _It=:, It_=:)
    metrics = Dict{String,Any}("completed" => false, "final_test_opened" => false,
        "representation_admitted" => false, "pde_comparator_admitted" => false,
        "exact_resume_supported" => false)
    try
        model, stats, metadata = train_SNF(
            file, joinpath(out, "native_model"), 32, 0, 5, 64, 128, 210;
            rng=MersenneTwister(2026111991), warmup=false,
            _batchsize=262144, batchsize_=262144,
            λ2=0.001, α=0.0, weight_decays=0.01,
            makedata_kws=split, verbose=true, device=Lux.gpu_device())
        @assert eltype(model[2]) === Float64
        @assert all(eltype(s) === Float64 for s in stats[2:end])
        # Native stage files and this final artifact deliberately do not pretend
        # to contain optimizer/RNG state for exact resumption.
        jldsave(joinpath(out, "native_model", "final_native_model.jld2"); model, STATS=stats, metadata)
        metrics["completed"] = true
        metrics["native_training_steps"] = NeuralROMs.B8_FLOAT64_AUDIT["steps"]
        metrics["gpu_float64_array_checks"] = NeuralROMs.B8_FLOAT64_AUDIT["arrays"]
        metrics["latent_dimension"] = 32
        metrics["hyper_input_dimension"] = 17
        metrics["decoder_sine_hidden_layers"] = 6
        metrics["train_batch_size"] = 262144
        metrics["validation_batch_size"] = 262144
    catch err
        metrics["error"] = sprint(showerror, err, catch_backtrace())
        rethrow()
    finally
        open(joinpath(out, "expanded-training.toml"), "w") do io
            TOML.print(io, metrics)
        end
    end
end
main()
