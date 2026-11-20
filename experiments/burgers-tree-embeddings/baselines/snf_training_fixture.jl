# Tiny training-path qualification only: no study or final-test solution fields.
using NeuralROMs, CUDA, Lux, LuxCUDA, ComponentArrays, Random, LinearAlgebra
using JLD2, TOML, Optimisers, Zygote, MLUtils, Statistics
@assert CUDA.functional()
CUDA.allowscalar(false)
out = ENV["B8_SNF_OUTPUT"]
@assert realpath(pkgdir(NeuralROMs)) == realpath(ENV["B8_SNF_PROJECT"])
include(joinpath(pkgdir(NeuralROMs), "experiments_SNFROM", "smoothNF.jl"))
BLAS.set_num_threads(4)

function main()
axis = collect(range(0.1, 0.9; length=4))
x = hcat(([xx, yy] for xx in axis for yy in axis)...)
t = [0.0, 0.125, 0.25]
mu = [[0.2], [0.5], [0.8]]
u = zeros(Float64, 1, size(x, 2), length(mu), length(t))
for j in axes(u, 3), k in axes(u, 4)
    u[1, :, j, k] .= (0.2+mu[j][1]) .* sin.(pi.*x[1, :]).^2 .* sin.(pi.*x[2, :]).^2 .* exp.(-t[k])
end
metadata = (; grid=(4, 4), role="synthetic native training-path qualification only")
original = joinpath(out, "tiny_training.jld2")
jldsave(original; x, u, t, mu, metadata)
shifted_u = copy(u); shifted_u[:, :, 3, :] .+= 100.0
shifted = joinpath(out, "tiny_shifted_development.jld2")
jldsave(shifted; x, u=shifted_u, t, mu, metadata)
split = (; Ix=:, _Ib=[1, 2], Ib_=[3], _It=:, It_=:)
train_a, dev_a, md_a = makedata_SNF(original; split..., verbose=false)
train_b, dev_b, md_b = makedata_SNF(shifted; split..., verbose=false)
for key in (:ū, :σu, :x̄, :σx, :t̄, :σt)
    @assert getproperty(md_a, key) == getproperty(md_b, key)
end
@assert train_a[1][1] == train_b[1][1]
@assert train_a[1][2] == train_b[1][2]
@assert train_a[2] == train_b[2]
@assert maximum(abs.(dev_a[2]-dev_b[2])) > 1.0
expected_mean = mean(u[:, :, 1:2, :])
@assert abs(only(md_a.ū)-expected_mean) < 1e-14
@assert all(eltype(v) === Float64 for v in (train_a[1]..., train_a[2], dev_a[1]..., dev_a[2]))

# E=7/warmup=false executes one tiny epoch in each of the author's seven
# optimizer stages. The architecture and regularized native loss stay intact.
native_model, stats, trained_metadata = train_SNF(
    original, joinpath(out, "tiny_native_model"), 2, 0, 1, 8, 16, 7;
    rng=MersenneTwister(2026111988), warmup=false,
    _batchsize=numobs(train_a), batchsize_=numobs(dev_a),
    λ2=0.001, α=0.0, weight_decays=0.01, makedata_kws=split,
    verbose=false, device=Lux.gpu_device())
@assert NeuralROMs.B8_FLOAT64_AUDIT["steps"] == 7
@assert NeuralROMs.B8_FLOAT64_AUDIT["arrays"] > 0
@assert all(eltype(s) === Float64 for s in stats[2:end])
NN, p_host, st_host = native_model
@assert eltype(p_host) === Float64
device = Lux.gpu_device()
# Preserve parameter values/axes without the Lux adaptor reconstructing GPU leaves.
p = ComponentArray(CUDA.CuArray(Float64.(getdata(p_host))), getaxes(p_host))
st = device(st_host)
@assert getaxes(p) == getaxes(p_host)
@assert Array(getdata(p)) == Float64.(getdata(p_host))
NeuralROMs.b8_assert_gpu_float64(st)
st = Lux.trainmode(st)
NeuralROMs.b8_assert_gpu_float64(p)
# CuIterator's pinned implementation preserves input Float64; copy the first
# batch so iterator reuse cannot free arrays retained by this small fixture.
loader = CUDA.CuIterator(MLUtils.DataLoader(train_a; batchsize=numobs(train_a), shuffle=false))
raw = first(loader)
batch = ((copy(raw[1][1]), copy(raw[1][2])), copy(raw[2]))
NeuralROMs.b8_assert_gpu_float64(batch)
lossfun = NeuralROMs.regularize_flatdecoder(NeuralROMs.mse; α=0.0, λ2=0.001)
loss = NeuralROMs.Loss(NN, st, batch, lossfun)
value, new_st, info, gradient = NeuralROMs.grad(loss, p)
@assert value isa Float64
NeuralROMs.b8_assert_gpu_float64(gradient)
direction = copy(p); fill!(direction, 1/sqrt(length(direction)))
step = 1e-6
plus = ComponentArray(getdata(p) .+ step.*getdata(direction), getaxes(p))
minus = ComponentArray(getdata(p) .- step.*getdata(direction), getaxes(p))
fd = (first(loss(plus))-first(loss(minus)))/(2*step)
ad = sum(gradient .* direction)
gradient_error = abs(fd-ad)/(1+abs(fd))
@assert gradient_error < 1e-6

# Two explicit native Loss/grad/Optimisers Adam updates test the real
# FlatDecoder gradient, not the previous B7 unrelated Lux Dense toy network.
rule = Optimisers.Adam(1e-5)
opt_state = Optimisers.setup(rule, p)
losses = Float64[value]
for iteration in 1:2
    current_loss = NeuralROMs.Loss(NN, st, batch, lossfun)
    l, st, _, g = NeuralROMs.grad(current_loss, p)
    NeuralROMs.b8_assert_gpu_float64(g)
    opt_state, p = Optimisers.update!(opt_state, p, g)
    NeuralROMs.b8_assert_gpu_float64(p)
    push!(losses, first(lossfun(NN, p, st, batch)))
end
@assert all(isfinite, losses) && losses[end] < losses[1]
@assert all(losses[2:end] .< losses[1:end-1])
metrics = Dict(
    "passed" => true,
    "training_only_normalization_invariant_to_development_shift" => true,
    "train_mean_absolute_error" => abs(only(md_a.ū)-expected_mean),
    "flat_parameter_transfer_preserves_values_and_axes" => true,
    "native_training_steps" => NeuralROMs.B8_FLOAT64_AUDIT["steps"],
    "gpu_float64_array_checks" => NeuralROMs.B8_FLOAT64_AUDIT["arrays"],
    "directional_gradient_relative_error" => gradient_error,
    "two_native_adam_step_losses" => losses,
    "trained_comparator" => false,
    "study_data_opened" => false,
    "final_test_opened" => false)
open(joinpath(out, "training-fixture.toml"), "w") do io
    TOML.print(io, metrics)
end
println("B8_NATIVE_SNF_TRAINING_PATH_PASS ", metrics)
end

main()
