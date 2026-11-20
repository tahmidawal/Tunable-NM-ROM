# Included by the retained native-training fixture. No study solution fields.
function b8_boundary_checks(native_model)
    NN, p, st = native_model
    wrapper = NN.layers.decoder
    @assert wrapper isa B8BoundaryDecoder
    # Delegation preserves native setup order, parameter axes, values and state.
    rng = MersenneTwister(97)
    pw, sw = Lux.setup(copy(rng), wrapper)
    pn, sn = Lux.setup(copy(rng), wrapper.native)
    aw, an = ComponentArray(pw), ComponentArray(pn)
    @assert getaxes(aw)==getaxes(an) && getdata(aw)==getdata(an) && sw==sn
    dp, ds = p.decoder, st.decoder
    latent = size(dp.layer_1.weight, 2)-2
    z = collect(range(-0.2, 0.3; length=latent))
    coordinates = [0.0 1.0 0.2 0.7 0.0 1.0 0.31 0.63;
                   0.3 0.6 0.0 1.0 0.0 1.0 0.42 0.28]
    function inputs(x, code)
        xn = (x .- collect(wrapper.xmean)) ./ collect(wrapper.xscale)
        vcat(xn, code)
    end
    function physical(x, code; wrapped=true)
        decoder = wrapped ? wrapper : wrapper.native
        normalized = only(first(decoder(reshape(inputs(x, code), :, 1), dp, ds)))
        wrapper.umean + wrapper.uscale*normalized
    end
    envelope(x) = 16*x[1]*(1-x[1])*x[2]*(1-x[2])
    envelope_grad(x) = [16*(1-2*x[1])*x[2]*(1-x[2]),16*x[1]*(1-x[1])*(1-2*x[2])]
    envelope_lap(x) = -32*(x[1]*(1-x[1])+x[2]*(1-x[2]))
    batch = hcat([inputs(coordinates[:, i],z) for i in axes(coordinates,2)]...)
    gpu_p = ComponentArray(CUDA.CuArray(Float64.(getdata(dp))), getaxes(dp))
    gpu_st = Lux.gpu_device()(ds)
    raw_gpu = first(wrapper.native(CUDA.CuArray(batch),gpu_p,gpu_st))
    wrapped_gpu = first(wrapper(CUDA.CuArray(batch),gpu_p,gpu_st))
    @assert eltype(wrapped_gpu)===Float64
    physical_gpu = Array(wrapper.umean .+ wrapper.uscale .* wrapped_gpu)
    raw_physical = Array(wrapper.umean .+ wrapper.uscale .* raw_gpu)
    b = reshape([envelope(coordinates[:,i]) for i in axes(coordinates,2)],1,:)
    identity_error = maximum(abs.(physical_gpu-b.*raw_physical))
    boundary_error = maximum(abs.(physical_gpu[:,1:6]))
    @assert identity_error<1e-12 && boundary_error<1e-12
    # Native train_model returns host parameters: these input-gradient/Hessian
    # checks are intentionally cluster CPU checks, not GPU nested-AD admission.
    gradient_error=0.0; laplacian_error=0.0; latent_fd_error=0.0
    for x in ([0.31,0.42],[0.63,0.28],[1/16384,0.43])
        raw = physical(x,z;wrapped=false)
        grad_raw = only(Zygote.gradient(xx->physical(xx,z;wrapped=false),x))
        grad_wrapped = only(Zygote.gradient(xx->physical(xx,z),x))
        expected = envelope(x).*grad_raw .+ raw.*envelope_grad(x)
        gradient_error=max(gradient_error,maximum(abs.(grad_wrapped-expected))/(1+maximum(abs.(expected))))
        lap_raw=tr(Zygote.hessian(xx->physical(xx,z;wrapped=false),x))
        lap_wrapped=tr(Zygote.hessian(xx->physical(xx,z),x))
        lap_expected=envelope(x)*lap_raw+2*dot(envelope_grad(x),grad_raw)+raw*envelope_lap(x)
        laplacian_error=max(laplacian_error,abs(lap_wrapped-lap_expected)/(1+abs(lap_expected)))
        gz=only(Zygote.gradient(code->physical(x,code),z))
        for j in eachindex(z)
            delta=zeros(Float64,latent);delta[j]=1e-6
            fd=(physical(x,z+delta)-physical(x,z-delta))/(2e-6)
            latent_fd_error=max(latent_fd_error,abs(fd-gz[j])/(1+abs(fd)))
        end
    end
    @assert gradient_error<1e-10 && laplacian_error<1e-9 && latent_fd_error<1e-6
    Dict("boundary_wrapper_passed"=>true,
        "value_and_native_training_gradient_device"=>"GPU Float64",
        "spatial_latent_gradient_and_hessian_device"=>"cluster CPU Float64",
        "gpu_nested_spatial_ad_qualified"=>false,
        "online_derivative_path_qualification_required"=>true,"boundary_max_absolute"=>boundary_error,
        "physical_normalization_identity_error"=>identity_error,
        "spatial_product_rule_relative_error"=>gradient_error,
        "laplacian_product_rule_relative_error"=>laplacian_error,
        "latent_gradient_fd_relative_error"=>latent_fd_error,
        "native_parameter_axes_values_and_state_preserved"=>true,
        "classification"=>"Continuous homogeneous-Dirichlet new-PDE port; not unchanged published SNF")
end
