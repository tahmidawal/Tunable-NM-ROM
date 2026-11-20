# Continuous physical Dirichlet port; native parameters and state are unchanged.
# Decoder input ordering is [normalized physical coordinates; latent code].
struct B8BoundaryDecoder{D} <: Lux.AbstractExplicitLayer
    native::D
    xmean::NTuple{2,Float64}
    xscale::NTuple{2,Float64}
    umean::Float64
    uscale::Float64
end
function B8BoundaryDecoder(native, xmean, xscale, umean, uscale)
    @assert length(xmean)==length(xscale)==2
    @assert length(umean)==length(uscale)==1
    @assert all(isfinite, xmean) && all(isfinite, xscale) && all(xscale .> 0)
    @assert isfinite(only(umean)) && isfinite(only(uscale)) && only(uscale)>0
    B8BoundaryDecoder{typeof(native)}(native, Tuple(Float64.(xmean)), Tuple(Float64.(xscale)),
        Float64(only(umean)), Float64(only(uscale)))
end
Lux.initialparameters(rng::Random.AbstractRNG, model::B8BoundaryDecoder) = Lux.initialparameters(rng, model.native)
Lux.initialstates(rng::Random.AbstractRNG, model::B8BoundaryDecoder) = Lux.initialstates(rng, model.native)
function (model::B8BoundaryDecoder)(input::AbstractMatrix, p, st)
    raw, next_state = model.native(input, p, st)
    # Keep geometry scalar/tuple constants; never scalar-index a GPU array.
    x = input[1:1, :] .* model.xscale[1] .+ model.xmean[1]
    y = input[2:2, :] .* model.xscale[2] .+ model.xmean[2]
    b = 16 .* x .* (1 .- x) .* y .* (1 .- y)
    # Denormalizing this expression gives b*(umean + uscale*raw).
    normalized = b .* raw .+ (b .- 1) .* (model.umean/model.uscale)
    normalized, next_state
end
