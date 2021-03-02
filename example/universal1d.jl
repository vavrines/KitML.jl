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

cd(@__DIR__)
include("math.jl")

# moments
L = 1
ne = (L + 1)^2
#m = KitBase.eval_spherharmonic(points, L)
m = ComputeSphericalBasisAnalytical(points)