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

(nn::BGKNet)(x, p, vs = VSpace1D(-6, 6, 40; precision = Float32), γ = 3) = begin
    np1 = param_length(nn.Mnet)
    M = f_maxwellian(x)
    nn.νnet(x, p[np1+1:end]) .* (nn.fn(M .+ nn.Mnet(x, p[1:np1]), x))
end

Solaris.init_params(nn::BGKNet) = vcat(init_params(nn.Mnet), init_params(nn.νnet))
