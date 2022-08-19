# ============================================================
# Neural Network Models
# ============================================================

export BGKNet

"""
$(TYPEDEF)

BGK relaxation network

# Fields

$(FIELDS)

# Forward pass

`(nn::BGKNet)(x, p, vs = VSpace1D(-6, 6, size(x)[1] - 1; precision = Float32), γ = 3) `

Last row of x is set as mean relaxation time
"""
struct BGKNet{T1,T2}
    Mnet::T1
    νnet::T2
end

(nn::BGKNet)(x, p, vs = VSpace1D(-6, 6, size(x)[1] - 1; precision = Float32), γ = 3) = begin
    nm = param_length(nn.Mnet)
    f = @view x[begin:end-1, :]
    τ = @view x[end:end, :]
    M = f_maxwellian(f, vs, γ)
    y = f .- M
    z = vcat(y, τ)
    
    (relu(M .+ nn.Mnet(y, p[1:nm])) .- f) ./ (τ .* (1 .+ 0.2 .* tanh.(nn.νnet(z, p[nm+1:end]))))
end

Solaris.init_params(nn::BGKNet) = vcat(init_params(nn.Mnet), init_params(nn.νnet))
