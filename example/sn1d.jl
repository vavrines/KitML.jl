using KitBase, KitBase.FastGaussQuadrature
using ProgressMeter, Plots

begin
    # case
    matter = "photon"
    case = "linesource"
    space = "1d1f1v"
    nSpecies = 1
    flux = "kfvs"
    collision = "bgk"
    interpOrder = 2
    limiter = "vanleer"
    boundary = "fix"
    cfl = 0.5
    maxTime = 0.2

    # physical space
    x0 = 0
    x1 = 1
    nx = 100
    pMeshType = "uniform"
    nxg = 0

    # velocity space
    vMeshType = "rectangle"
    umin = -1
    umax = 1
    nu = 28
    nug = 0

    # gas
    knudsen = 0.01
    mach = 0.0
    prandtl = 1
    inK = 0
    omega = 0.5
    alphaRef = 1.0
    omegaRef = 0.5
end

set = Setup(
    matter,
    case,
    space,
    flux,
    collision,
    nSpecies,
    interpOrder,
    limiter,
    boundary,
    cfl,
    maxTime,
)

pSpace = PSpace1D(x0, x1, nx, nxg)

points, weights = gausslegendre(nu)
vSpace = VSpace1D(points[1], points[end], nu, points, ones(nu).*(points[end]-points[1])/(nu-1), weights)

μᵣ = ref_vhs_vis(knudsen, alphaRef, omegaRef)
gas = Gas(knudsen, mach, prandtl, inK, 3.0, omega, alphaRef, omegaRef, μᵣ)

fL = 0.5 * ones(nu)
wL = moments_conserve(fL, points, weights)
primL = conserve_prim(wL, 3.0)
fR = 0.0001 * ones(nu)
wR = moments_conserve(fR, points, weights)
primR = conserve_prim(wR, 3.0)
ib = IB1F(wL, primL, fL, primL, wR, primR, fR, primR)

ks = SolverSet(set, pSpace, vSpace, gas, ib, @__DIR__)

ctr = Array{ControlVolume1D1F}(undef, ks.pSpace.nx)
face = Array{Interface1D1F}(undef, ks.pSpace.nx + 1)
for i in eachindex(ctr)
    ctr[i] = ControlVolume1D1F(
        ks.pSpace.x[i],
        ks.pSpace.dx[i],
        ks.ib.wR,
        ks.ib.primR,
        ks.ib.fR,
    )
end

for i = 1:ks.pSpace.nx+1
    face[i] = Interface1D1F(ks.ib.wL, ks.ib.fL)
end

function fboundary(
    ff::T2,
    f::T4,
    u::T5,
    dt,
    rot = 1,
) where {
    T2<:AbstractArray{<:AbstractFloat,1},
    T4<:AbstractArray{<:AbstractFloat,1},
    T5<:AbstractArray{<:AbstractFloat,1},
}

    δ = heaviside.(u .* rot)

    fWall = 0.5 .* δ .+ f .* (1.0 .- δ)

    @. ff = u * fWall * dt

    return nothing

end

function step(
    ffL::T2,
    w::T3,
    f::T4,
    ffR::T2,
    u::T5,
    weights::T5,
    τ,
    dx,
    dt,
) where {
    T2<:AbstractArray{<:AbstractFloat,1},
    T3<:AbstractArray{<:AbstractFloat,1},
    T4<:AbstractArray{<:AbstractFloat,1},
    T5<:AbstractArray{<:AbstractFloat,1},
}
    M = sum(weights .* f)
    for i in eachindex(u)
        f[i] += (ffL[i] - ffR[i]) / dx + (M - f[i]) / τ * dt
    end
    w[1] = sum(weights .* f)
end

dt = cfl * pSpace.dx[1] 
nt = maxTime / dt |> floor |> Int

anim = @animate for iter = 1:150#nt
    #reconstruct!(ks, ctr)

    fboundary(
        face[1].ff,
        ctr[1].f,
        vSpace.u,
        dt,
    )
    @inbounds for i = 2:nx
        flux_kfvs!(
            face[i].ff,
            ctr[i-1].f,
            ctr[i].f,
            vSpace.u,
            dt,
            ctr[i-1].sf,
            ctr[i].sf,
        )
    end

    @inbounds for i = 1:nx-1
        step(
            face[i].ff,
            ctr[i].w,
            ctr[i].f,
            face[i+1].ff,
            vSpace.u,
            vSpace.weights,
            1.0,
            pSpace.dx[i],
            dt,
        )
    end

    pltx = ks.pSpace.x[1:ks.pSpace.nx]
    plty = zeros(ks.pSpace.nx, 6)
    for i in eachindex(pltx)
        plty[i, 1] = ctr[i].w[1]
    end
    plot(pltx, plty[:, 1], label = "Density", lw = 2, xlabel = "x")
end

gif(anim, "propagation.gif")


begin
    pltx = ks.pSpace.x[1:ks.pSpace.nx]
    plty = zeros(ks.pSpace.nx, 6)
    for i in eachindex(pltx)
        plty[i, 1] = ctr[i].w[1]
    end
    plot(pltx, plty[:, 1], label = "Density", lw = 2, xlabel = "x")
end