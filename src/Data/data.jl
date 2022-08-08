# ============================================================
# Data Setup & Generation
# ============================================================

export regime_data

"""
$(SIGNATURES)

Generate dataset for fluid regime classification
"""
function regime_data(args...)
    if length(args[1]) == 3
        regime_data_1d(args...)
    elseif length(args[1]) == 4
        regime_data_2d(args...)
    elseif length(args[1]) == 5
        regime_data_3d(args...)
    end
end

"""
$(SIGNATURES)
"""
function regime_data(ks::SolverSet, args...)
    if length(args[1]) == 3
        regime_data_1d(ks, args...)
    elseif length(args[1]) == 4
        regime_data_2d(ks, args...)
    elseif length(args[1]) == 5
        regime_data_3d(ks, args...)
    end
end

function regime_data_1d(
    w,
    sw,
    f,
    u,
    K::Real,
    Kn::Real,
    μ = ref_vhs_vis(Kn, 1.0, 0.5),
    ω = 0.81,
)
    γ = heat_capacity_ratio(K, 1)
    prim = conserve_prim(w, γ)

    Mu, Mxi, _, _1 = gauss_moments(prim, K)
    a = pdf_slope(prim, sw, K)
    swt = -prim[1] .* moments_conserve_slope(a, Mu, Mxi, 1)
    A = pdf_slope(prim, swt, K)
    tau = vhs_collision_time(prim, μ, ω)
    fr = chapman_enskog(u, prim, a, A, tau)
    L = norm((f .- fr) ./ prim[1])

    x = [w; sw; tau]
    y = ifelse(L <= 0.005, 0.0, 1.0)
    return x, y
end

regime_data_1d(ks::SolverSet, w, sw, f) =
    regime_data(w, sw, f, ks.vs.u, ks.gas.K, ks.gas.Kn)

function regime_data_2d(
    w,
    swx,
    swy,
    f,
    u,
    v,
    K::Real,
    Kn::Real,
    μ = ref_vhs_vis(Kn, 1.0, 0.5),
    ω = 0.81,
)
    γ = heat_capacity_ratio(K, 2)
    prim = conserve_prim(w, γ)

    Mu, Mv, Mxi, _, _1 = gauss_moments(prim, K)
    a = pdf_slope(prim, swx, K)
    b = pdf_slope(prim, swy, K)
    swt =
        -prim[1] .* (
            moments_conserve_slope(a, Mu, Mv, Mxi, 1, 0) .+
            moments_conserve_slope(b, Mu, Mv, Mxi, 0, 1)
        )
    A = pdf_slope(prim, swt, K)
    tau = vhs_collision_time(prim, μ, ω)
    fr = chapman_enskog(u, v, prim, a, b, A, tau)
    L = norm((f .- fr) ./ prim[1])

    sw = (swx .^ 2 + swy .^ 2) .^ 0.5
    x = [w; sw; tau]
    y = ifelse(L <= 0.005, 0.0, 1.0)

    return x, y
end

regime_data_2d(ks::SolverSet, w, swx, swy, f) =
    regime_data(w, swx, swy, f, ks.vs.u, ks.vs.v, ks.gas.K, ks.gas.Kn)

function regime_data_3d(
    cons,
    swx,
    swy,
    swz,
    f,
    u,
    v,
    w,
    K::Real,
    Kn::Real,
    μ = ref_vhs_vis(Kn, 1.0, 0.5),
    ω = 0.81,
)
    γ = heat_capacity_ratio(K, 3)
    prim = conserve_prim(cons, γ)

    Mu, Mv, Mw, _, _1 = gauss_moments(prim, K)
    a = pdf_slope(prim, swx, K)
    b = pdf_slope(prim, swy, K)
    c = pdf_slope(prim, swz, K)
    swt =
        -prim[1] .* (
            moments_conserve_slope(a, Mu, Mv, Mw, 1, 0, 0) .+
            moments_conserve_slope(b, Mu, Mv, Mw, 0, 1, 0) .+
            moments_conserve_slope(c, Mu, Mv, Mw, 0, 0, 1)
        )
    A = pdf_slope(prim, swt, K)
    tau = vhs_collision_time(prim, μ, ω)
    fr = chapman_enskog(u, v, w, prim, a, b, c, A, tau)
    L = norm((f .- fr) ./ prim[1])

    sw = (swx .^ 2 + swy .^ 2 + swz .^ 2) .^ 0.5
    x = [cons; sw; tau]
    y = ifelse(L <= 0.005, 0.0, 1.0)

    return x, y
end

regime_data_3d(ks::SolverSet, w, swx, swy, swz, f) =
    regime_data(w, swx, swy, swz, f, ks.vs.u, ks.vs.v, ks.vs.w, ks.gas.K, ks.gas.Kn)
