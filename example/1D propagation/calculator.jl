using Flux, DiffEqFlux, DiffEqFlux.GalacticOptim, Plots, LinearAlgebra, Optim
using KitBase, KitBase.FastGaussQuadrature, KitBase.ProgressMeter
import KitML

function flux_wall!(
    ff::T1,
    f::T2,
    u::T3,
    dt,
    rot = 1,
) where {
    T1<:AbstractVector{<:AbstractFloat},
    T2<:AbstractVector{<:AbstractFloat},
    T3<:AbstractVector{<:AbstractFloat},
}
    δ = heaviside.(u .* rot)
    fWall = 0.5 .* δ .+ f .* (1.0 .- δ)
    @. ff = u * fWall * dt

    return nothing
end

begin
    # setup
    set = Setup("radiation", "linesource", "1d1f1v", "kfvs", "bgk", 1, 2, "vanleer", "extra", 0.3, 0.8)

    # physical space
    x0 = 0
    x1 = 1
    nx = 100
    nxg = 0
    ps = PSpace1D(x0, x1, nx, nxg)

    # velocity space
    nu = 28
    points, weights = gausslegendre(nu)
    vs = VSpace1D(points[1], points[end], nu, points, ones(nu) .* (points[end] - points[1]) / (nu - 1), weights)

    # material
    σs = ones(Float32, nx)
    σa = zeros(Float32, nx)
    σt = σs + σa
    σq = zeros(Float32, nx)

    # moments
    L = 1
    ne = 2
    m = eval_sphermonomial(points, L)

    # time
    cfl = 0.5
    dt = cfl * ps.dx[1]
    nt = set.maxTime / dt |> floor |> Int
    global t = 0.0

    # solution
    f0 = 0.0001 * ones(nu)
    phi = zeros(ne, nx)
    for i = 1:nx
        phi[:, i] .= m * f0
    end
    α = zeros(Float32, ne, nx)

    # NN
    αT = zeros(Float32, nx, ne)
    phiT = zeros(Float32, nx, ne)
    phi_old = zeros(Float32, ne, nx)
    phi_temp = deepcopy(phi_old)
    
    opt = optimize_closure(zeros(Float32, 2), m, weights, phi[:, 1], KitBase.maxwell_boltzmann_dual)
    
    global X = zeros(Float32, ne, 1)
    X[:, 1] .= phi[:, 1]
    global Y = zeros(Float32, 1, 1) # h
    Y[1, 1] = kinetic_entropy(opt.minimizer, m, weights)
    #global Y = zeros(Float32, ne, 1) # α
    #Y[:, 1] = opt.minimizer
end

begin
    # initial condition
    f0 = 0.0001 * ones(nu)
    phi = zeros(ne, nx)
    for i = 1:nx
        phi[:, i] .= m * f0
    end
    α = zeros(Float32, ne, nx)
    flux = zeros(Float32, ne, nx + 1)
    fη = zeros(nu)
end

#anim = @animate 
for iter = 1:nt
    println("iteration $iter of $nt")

    # mathematical optimizer
    @inbounds for i = 1:nx
        opt = KitBase.optimize_closure(α[:, i], m, weights, phi[:, i], KitBase.maxwell_boltzmann_dual)
        α[:, i] .= opt.minimizer
        phi[:, i] .= KitBase.realizable_reconstruct(opt.minimizer, m, weights, KitBase.maxwell_boltzmann_dual_prime)
    
        #X = hcat(X, phi[:, i])
        #Y = hcat(Y, kinetic_entropy(α[:, i], m, weights))
    end

    flux_wall!(fη, maxwell_boltzmann_dual.(α[:, 1]' * m)[:], points, dt, 1.0)
    for k in axes(flux, 1)
        flux[k, 1] = sum(m[k, :] .* weights .* fη)
    end

    @inbounds for i = 2:nx
        KitBase.flux_kfvs!(fη, KitBase.maxwell_boltzmann_dual.(α[:, i-1]' * m)[:], KitBase.maxwell_boltzmann_dual.(α[:, i]' * m)[:], points, dt)
        
        for k in axes(flux, 1)
            flux[k, i] = sum(m[k, :] .* weights .* fη)
        end
    end

    @inbounds for i = 1:nx-1
        for q = 1:1
            phi[q, i] =
                phi[q, i] +
                (flux[q, i] - flux[q, i+1]) / ps.dx[i] +
                (σs[i] * phi[q, i] - σt[i] * phi[q, i]) * dt +
                σq[i] * dt
        end

        for q = 2:ne
            phi[q, i] =
                phi[q, i] +
                (flux[q, i] - flux[q, i+1]) / ps.dx[i] +
                (-σt[i] * phi[q, i]) * dt
        end
    end
    phi[:, nx] .=  phi[:, nx-1]

    global t += dt

    #plot(ps.x[1:nx], phi[1, :])
end
#gif(anim, "newton1d.gif")

for i = 1:nx
    X = hcat(X, phi[:, i])
    Y = hcat(Y, kinetic_entropy(α[:, i], m, weights))
    #Y = hcat(Y, α[:, i])
end

fnn = FastChain(FastDense(2, 10, tanh), FastDense(10, 10, tanh), FastDense(10, 10, tanh), FastDense(10, 1))
res = KitML.sci_train(fnn, (X, Y); maxiters = 2000)
res = KitML.sci_train(fnn, (X, Y), res.u, LBFGS(); maxiters=1000)

y = fnn(phi, res.u)
_α = KitML.neural_closure(fnn, res.u, phi)

_α = zeros(Float32, ne, nx)
for j = 1:nx
    _α[:, j] .= KitML.neural_closure(fnn, res.u, phi[:, j])
end

plot(Y[1, 2:end][:], label="h (exact)")
scatter!(y[:], label="h (nn)")
plot!(permutedims(α), label="alpha (exact)")
scatter!(permutedims(_α), label="alpha (nn)")

h_recons = zeros(Float32, nx)
for i = 1:nx
    h_recons[i] = kinetic_entropy(_α[:, i], m, weights)
end
plot(Y[1, 2:end][:], legend=:none)
scatter!(h_recons)
