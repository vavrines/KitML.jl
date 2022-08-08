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

Forward pass of BGK relaxation network

Last row of x is set as reference viscosity
"""
(nn::BGKNet)(x, p, vs = VSpace1D(-6, 6, size(x)[1] - 1; precision = Float32), γ = 3) = begin
    nm = param_length(nn.Mnet)
    f = @view x[begin:end-1, :] # μ = @view x[end, :]
    M = f_maxwellian(f, vs, γ)
    nn.νnet(x, p[nm+1:end]) .* (nn.fn(M .+ nn.Mnet(f, p[1:nm]), f))
end

Solaris.init_params(nn::BGKNet) = vcat(init_params(nn.Mnet), init_params(nn.νnet))
