# ============================================================
# Neural Network Models
# ============================================================

export BGKNet

"""
$(TYPEDEF)

BGK relaxation network

# Fields

$(FIELDS)
"""
struct BGKNet{T1,T2,T3}
    Mnet::T1
    νnet::T2
    fn::T3
end

BGKNet(m, ν) = BGKNet(m, ν, -)

"""
$(SIGNATURES)

Forward pass of BGK network

Last row of x is set as mean relaxation time
"""
(nn::BGKNet)(x, p, vs = VSpace1D(-6, 6, size(x)[1] - 1; precision = Float32), γ = 3) = begin
    nm = param_length(nn.Mnet)
    f = @view x[begin:end-1, :]
    τ = @view x[end:end, :]
    M = f_maxwellian(f, vs, γ)
    y = f .- M
    z = vcat(y, τ)
    
    (nn.fn(relu(M .+ nn.Mnet(y, p[1:nm])), f)) ./ relu(τ .+ nn.νnet(z, p[nm+1:end]))
end

Solaris.init_params(nn::BGKNet) = vcat(init_params(nn.Mnet), init_params(nn.νnet))
